import os
import streamlit as st
import google.generativeai as genai
from openai import AzureOpenAI
from qdrant_client import QdrantClient

# --- CONFIGURATION ---

from config import (
    GOOGLE_API_KEY, GOOGLE_EMBED_MODEL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_EMBED_DEPLOYMENT, AZURE_CHAT_DEPLOYMENT,
    CHUNK_SIZES, QDRANT_CONFIGS
)

genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """
You are a farming knowledge synthesis expert. Your task is to take knowledge chunks and synthesize them into concise, detailed answers in natural conversational Telugu.
ABSOLUTE RULES:
1. Keep answer to approximately 100 words
2. Include varieties, methods, fertilizer schedule
3. Telugu words for numbers (1.5 kg -> ఒకటిన్నర కిలోలు)
4. Natural conversational Telugu - NOT formal
5. No English words, no markdown headings, no bullet points
6. Synthesize chunks into one flowing answer
7. No greetings or preamble
""".strip()

def calculate_cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = sum(a * a for a in v1) ** 0.5
    norm_b = sum(b * b for b in v2) ** 0.5
    if not norm_a or not norm_b: return 0.0
    return dot_product / (norm_a * norm_b)

@st.cache_resource
def init_azure_client():
    return AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_version=AZURE_OPENAI_API_VERSION)

@st.cache_resource
def init_qdrant(methodology="semantic"):
    cfg = QDRANT_CONFIGS.get(methodology, QDRANT_CONFIGS["semantic"])
    return QdrantClient(url=cfg["url"], api_key=cfg["api_key"], timeout=60)

def build_context(chunks: list[dict]) -> str:
    return "\n\n".join([f"ముక్క {i+1}: {c['text']}" for i, c in enumerate(chunks)])

def get_google_embedding(text):
    result = genai.embed_content(model=GOOGLE_EMBED_MODEL, content=text, task_type="retrieval_query")
    return result['embedding']

def get_openai_embedding(azure_client, text):
    response = azure_client.embeddings.create(model=AZURE_EMBED_DEPLOYMENT, input=[text])
    return response.data[0].embedding

def retrieve_chunks_by_size(question: str, chunk_size: int, top_k: int, query_vector=None, collection_prefix="rnf_A_semantic_collection", methodology="semantic"):
    qdrant = init_qdrant(methodology)
    collection_name = f"{collection_prefix}_{chunk_size}"
    
    # Dimension safety
    if query_vector is not None and len(query_vector) == 3072:
        # Fallback to OpenAI collection if vector is 3072 dims and prefix doesn't match
        pass 

    try:
        results = qdrant.query_points(collection_name=collection_name, query=query_vector, limit=top_k, with_vectors=True, with_payload=True).points
        chunks = []
        for hit in results:
            chunks.append({
                "text": hit.payload.get("text", "") if hit.payload else "",
                "qdrant_score": hit.score,
                "cosine_similarity": calculate_cosine_similarity(query_vector, hit.vector) if hit.vector else hit.score
            })
        return chunks
    except Exception:
        return []

def retrieve_all_chunk_sizes(question: str, top_k: int, provider="openai", methodology="semantic", **kwargs):
    azure_client = init_azure_client()
    query_vector = None
    
    # Prefix mapping: semantic, neural, recursive, slumber
    # rnf_A_*      = OpenAI (text-embedding-3-large) embeddings of rnf_A.txt
    # rnf_google_* = Google (gemini-embedding-2) embeddings of rnf_A.txt
    prefix_map = {
        "openai": {
            "semantic":  "rnf_A_semantic_collection",
            "neural":    "rnf_A_neural_collection",
            "recursive": "rnf_A_collection",
            "slumber":   "rnf_A_slumber_collection",
        },
        "google": {
            "semantic":  "rnf_google_semantic_collection",
            "neural":    "rnf_google_neural_collection",
            "recursive": "rnf_google_collection",
            "slumber":   "rnf_google_slumber_collection",
        }
    }

    if provider == "google":
        try:
            query_vector = get_google_embedding(question)
            prefix = prefix_map["google"].get(methodology, "rnf_google_collection")
        except Exception as e:
            st.warning(f"⚠️ Google API error ({e}). Falling back to OpenAI embeddings.")
            query_vector = get_openai_embedding(azure_client, question)
            prefix = prefix_map["openai"].get(methodology, "rnf_A_collection")
    else:
        query_vector = get_openai_embedding(azure_client, question)
        prefix = prefix_map["openai"].get(methodology, "rnf_A_collection")

    results = {}
    for chunk_size in CHUNK_SIZES:
        results[chunk_size] = retrieve_chunks_by_size(
            question, chunk_size, top_k, query_vector=query_vector, 
            collection_prefix=prefix, methodology=methodology
        )
    
    return results, azure_client

def generate_answers_all_sizes(question: str, chunks_by_size: dict, azure_client: AzureOpenAI) -> dict:
    answers = {}
    for chunk_size in CHUNK_SIZES:
        chunks = chunks_by_size.get(chunk_size, [])
        context = build_context(chunks)
        
        if not chunks:
            answer = "సరిపోయే సమాచారం దొరకలేదు."
        else:
            completion = azure_client.chat.completions.create(
                model=AZURE_CHAT_DEPLOYMENT,
                temperature=0.7,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"ప్రశ్న: {question}\n\nజ్ఞాన ముక్కలు:\n{context}"}
                ]
            )
            answer = completion.choices[0].message.content.strip()
        
        top_3 = chunks[:3]
        precision = sum(c["cosine_similarity"] for c in top_3) / len(top_3) if top_3 else 0
        answers[chunk_size] = {"answer": answer, "chunks": chunks, "precision_at_3": precision, "num_chunks": len(chunks)}
    return answers
