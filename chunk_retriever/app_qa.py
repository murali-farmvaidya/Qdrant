import streamlit as st
import time
import re
import sys
import os
import requests
import json
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rag_utils
from config import QDRANT_CONFIGS, GOOGLE_API_KEY, GOOGLE_EMBED_MODEL
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from bm25 import BM25Okapi
from qdrant_client import QdrantClient

# Centralized Vertex Configuration
PROJECT_ID = os.getenv("VERTEX_SA_PROJECT_ID", "vertex-ai-project-494805")
LOCATION = "asia-south1"
MODEL_ID = "gemini-2.5-flash"

# Centralized System Prompt Rules requested by user
QA_SYSTEM_PROMPT = """
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

@st.cache_resource
def get_vertex_token():
    # Priority 1: Load directly from vertex_sa.json if available
    sa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vertex_sa.json'))
    if os.path.exists(sa_path):
        try:
            with open(sa_path, "r", encoding="utf-8") as f:
                sa_info = json.load(f)
            creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            return creds
        except Exception:
            pass
            
    # Priority 2: Fall back to env variables
    private_key = os.getenv("VERTEX_SA_PRIVATE_KEY")
    if private_key:
        private_key = private_key.strip('"').replace("\\n", "\n")
        
    info = {
        "type": "service_account",
        "project_id": PROJECT_ID,
        "private_key_id": os.getenv("VERTEX_SA_PRIVATE_KEY_ID"),
        "private_key": private_key,
        "client_email": os.getenv("VERTEX_SA_CLIENT_EMAIL"),
        "client_id": os.getenv("VERTEX_SA_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('VERTEX_SA_CLIENT_EMAIL')}",
        "universe_domain": "googleapis.com"
    }
    creds = service_account.Credentials.from_service_account_info(
        info, 
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    return creds

def call_vertex_gemini(prompt, system_instruction=None):
    try:
        creds = get_vertex_token()
        if not creds.valid:
            creds.refresh(Request())
        
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
        
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 8192
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            return "Error: No candidates returned from Vertex AI."
        else:
            raise Exception(f"Vertex API returned {response.status_code}: {response.text}")
    except Exception as e:
        # Fallback seamlessly to Standard Google Generative AI Developer API using GOOGLE_API_KEY
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name='gemini-2.5-flash',
                    generation_config={"temperature": 0.7, "max_output_tokens": 8192},
                    system_instruction=system_instruction
                )
            else:
                model = genai.GenerativeModel(
                    model_name='gemini-2.5-flash',
                    generation_config={"temperature": 0.7, "max_output_tokens": 8192}
                )
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as fallback_err:
            return f"Exception in Vertex AI Call: {e} (Fallback also failed: {fallback_err})"

# Query Normalization Agent using Gemini 2.5 Flash
# Query Normalization Agent using Gemini 2.5 Flash
def normalize_query(query):
    prompt = (
        "Extract the core agricultural entities and search keywords from this question. "
        "Return ONLY space-separated keywords. You MUST return them in the same language "
        "as the input question (e.g., if the question is in Telugu, you MUST return them in Telugu).\n"
        f"Question: {query}"
    )
    norm = call_vertex_gemini(prompt)
    if norm and "Exception" not in norm and "Error" not in norm:
        return norm.strip()
    return query

# RRF scoring formula
def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

# Parse Q&A chunks directly from the local file
def load_qa_chunks():
    source_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rnf_q_a.txt'))
    if not os.path.exists(source_file):
        source_file = "rnf_q_a.txt"
    
    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = re.split(r"^Chunk \d+", content, flags=re.MULTILINE | re.IGNORECASE)
    
    chunks = []
    for rc in raw_chunks:
        rc = rc.strip()
        if not rc:
            continue
        lines = [line.strip() for line in rc.split("\n") if line.strip()]
        if not lines:
            continue
        chunks.append("\n".join(lines))
    return chunks

# Local BM25 Index builder for Q&A Chunks
@st.cache_resource
def get_qa_bm25_index():
    chunks = load_qa_chunks()
    tokenized_chunks = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized_chunks), chunks

# Fetch Q&A questions for UI autocomplete
def get_telugu_questions():
    chunks = load_qa_chunks()
    questions = []
    for chunk in chunks:
        lines = chunk.split("\n")
        if lines:
            first_line = lines[0].strip()
            # Clean chunk number/preamble if any
            first_line = re.sub(r'^\d+\.\d+\.?\s*', '', first_line)
            questions.append(first_line)
    return questions

# Direct vector retrieval from the new 'qa' collection on Qdrant Cloud
def retrieve_qa_vector_chunks(query_vector, provider="google", top_k=3):
    cfg = QDRANT_CONFIGS["qa"]
    qdrant_client = QdrantClient(url=cfg["url"], api_key=cfg["api_key"], timeout=60)
    collection_name = "rnf_qa_google_collection_512" if provider == "google" else "rnf_qa_collection_512"
    
    try:
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k * 2,
            with_vectors=False,
            with_payload=True
        ).points
        
        chunks = []
        for hit in results:
            chunks.append({
                "text": hit.payload.get("text", "") if hit.payload else "",
                "cosine_similarity": hit.score
            })
        return chunks
    except Exception as e:
        st.error(f"Error querying Qdrant Cloud: {e}")
        return []

# Execute the complete hybrid master flowchart pipeline
def process_qa_hybrid(norm_query, raw_query, provider, top_k):
    # Step 1: Vector Dense Retrieval
    if provider == "google":
        query_vector = rag_utils.get_google_embedding(norm_query)
    else:
        azure_client = rag_utils.init_azure_client()
        query_vector = rag_utils.get_openai_embedding(azure_client, norm_query)
        
    vector_chunks = retrieve_qa_vector_chunks(query_vector, provider, top_k)
    
    # Step 2: Local BM25 Sparse Retrieval
    bm25_index, raw_chunks = get_qa_bm25_index()
    tokenized_query = norm_query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k*2]
    bm25_results = [{"text": raw_chunks[i], "score": bm25_scores[i]} for i in top_bm25_idx]
    
    # Step 3: Reciprocal Rank Fusion (RRF)
    scores = {}
    bm25_ranks = {}
    vector_ranks = {}
    
    for rank, hit in enumerate(bm25_results):
        t = hit["text"]
        scores[t] = scores.get(t, 0) + rrf_score(rank + 1)
        bm25_ranks[t] = rank + 1
        
    for rank, hit in enumerate(vector_chunks):
        t = hit["text"]
        scores[t] = scores.get(t, 0) + rrf_score(rank + 1)
        vector_ranks[t] = rank + 1
        
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Step 4: Reranking & Top Chunks Selection
    min_vector_score = min([c["cosine_similarity"] for c in vector_chunks]) if vector_chunks else 0.3
    vector_scores = {c["text"]: c["cosine_similarity"] for c in vector_chunks}
    
    top_merged = []
    for m in merged[:top_k]:
        text = m[0]
        rrf = m[1]
        cos_sim = vector_scores.get(text, min_vector_score)
        top_merged.append({
            "text": text,
            "cosine_similarity": cos_sim,
            "rrf_score": rrf,
            "bm25_rank": bm25_ranks.get(text, "N/A"),
            "vector_rank": vector_ranks.get(text, "N/A")
        })
        
    # Step 5: Synthesis Generation
    start_time = time.perf_counter()
    context = "\n\n".join([f"Chunk {i+1}:\n{c['text']}" for i, c in enumerate(top_merged)])
    answer_prompt = f"Question: {raw_query}\n\nKnowledge Chunks:\n{context}"
    answer = call_vertex_gemini(answer_prompt, system_instruction=QA_SYSTEM_PROMPT)
    elapsed = time.perf_counter() - start_time
    
    precision = sum(c["cosine_similarity"] for c in top_merged) / len(top_merged) if top_merged else 0.0
    
    return {
        "answer": answer,
        "chunks": top_merged,
        "precision": precision,
        "elapsed_time": elapsed,
        "vector_count": len(vector_chunks),
        "bm25_count": len(bm25_results)
    }

# ─── STREAMLIT LAYOUT ────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Q&A Structuring & Validation", page_icon="🌾", layout="wide")

# Custom Sleek CSS Styles
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1f29;
        border-radius: 8px;
        color: #cbd5e1;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
    .flowchart-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #4338ca;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
    }
    .flowchart-step {
        display: inline-block;
        background-color: #312e81;
        border: 1px solid #6366f1;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px;
        font-size: 14px;
        font-weight: 500;
        color: #e0e7ff;
    }
    .arrow {
        display: inline-block;
        font-size: 18px;
        color: #818cf8;
        padding: 0 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Telugu RAG Q&A Structuring & Validation")
st.caption("Centralized Benchmarking Pipeline for pre-processed Telugu Question & Answer chunks (63 pairs)")

# Flowchart UI rendering
with st.container():
    st.markdown('<div class="flowchart-box">', unsafe_allow_html=True)
    st.markdown("### 🏃 Active Master RAG Pipeline Flowchart")
    
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <div class="flowchart-step">1. User Query Input</div>
        <div class="arrow">➔</div>
        <div class="flowchart-step">2. Normalization Agent (Gemini 2.5 Flash)</div>
        <div class="arrow">➔</div>
        <div class="flowchart-step">3. Dual Retrieval (Sparse BM25 + Qdrant Dense Vector)</div>
        <div class="arrow">➔</div>
        <div class="flowchart-step">4. Reciprocal Rank Fusion (RRF)</div>
        <div class="arrow">➔</div>
        <div class="flowchart-step">5. Reranker (Top Chunks Selection)</div>
        <div class="arrow">➔</div>
        <div class="flowchart-step">6. Vertex AI Mumbai (Gemini 2.5 Flash Synthesizer)</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Central Sidebar Configs
with st.sidebar:
    st.header("⚙️ Pipeline Setup")
    
    provider = st.selectbox(
        "Select Embedding Model",
        ["Google (gemini-embedding-2)", "OpenAI (text-embedding-3-large)"],
        index=0
    )
    provider_key = "google" if "Google" in provider else "openai"
    
    top_k = st.slider("Select Top K Reference Chunks", 1, 10, 3)
    
    st.divider()
    st.markdown("### 📊 Active Cluster Configuration")
    st.markdown(f"**Host:** `{QDRANT_CONFIGS['qa']['url']}`")
    collection_name = "rnf_qa_google_collection_512" if provider_key == "google" else "rnf_qa_collection_512"
    st.markdown(f"**Target Collection:** `{collection_name}`")
    st.markdown("**Embedding Output Dimension:** `3072`")
    st.markdown("**Synthesis Engine:** `Vertex AI asia-south1 (Mumbai)`")
    st.markdown("**Base LLM:** `gemini-2.5-flash`")

# Auto-complete or paste Q&A Questions
st.subheader("📤 Question Selection & Input")
questions = get_telugu_questions()

selected_qa = st.selectbox(
    "Choose from one of the 63 test questions in the dataset:",
    ["-- Select Question --"] + questions
)

user_query = st.text_input(
    "OR Write a custom query in Telugu / English:",
    value=selected_qa if selected_qa != "-- Select Question --" else "",
    placeholder="ఉదాహరణకు: ఆచ్ఛాదన అంటే ఏమిటి?"
)

if st.button("🚀 Execute Pipeline Validation"):
    if not user_query.strip():
        st.warning("Please select or enter a query to run the RAG pipeline.")
    else:
        st.write("---")
        
        # Pipeline status indicators
        status_box = st.empty()
        
        with status_box.container():
            st.info("Step 1: Raw User Query received.")
            
        time.sleep(0.5)
        
        # Step 2: Normalization
        with status_box.container():
            st.warning("Step 2: Activating Normalization Agent...")
        norm_query = normalize_query(user_query)
        
        time.sleep(0.5)
        
        # Step 3: Retrieval, Fusion, Reranking, Synthesis
        with status_box.container():
            st.success(f"Step 2 Completed. Extracted Keywords: '{norm_query}'")
            st.warning("Step 3 & 4: Performing Local BM25 + Qdrant Cloud Vector Search & Fusion (RRF)...")
            
        pipeline_results = process_qa_hybrid(norm_query, user_query, provider_key, top_k)
        
        status_box.empty()
        
        # Display validation tabs
        tab_answer, tab_chunks, tab_logs = st.tabs([
            "💬 Synthesized Answer (Vertex AI)",
            "🔍 Fetched Chunks & Ranks",
            "🏃 Normalization & Telemetry"
        ])
        
        with tab_answer:
            st.markdown("### Synthesized Telugu Answer (Vertex AI Mumbai)")
            st.markdown(f"**Synthesized answer is generated under strict rules: natural conversational Telugu, ~100 words, no English words.**")
            
            st.info(pipeline_results["answer"])
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("🎯 Semantic Precision@3", f"{pipeline_results['precision']:.4f}")
            with c2:
                st.metric("⏱️ Execution Latency", f"{pipeline_results['elapsed_time']:.2f}s")
            with c3:
                st.metric("📚 References Provided", f"{len(pipeline_results['chunks'])}")
                
        with tab_chunks:
            st.markdown("### Retrieved Q&A Chunks")
            st.caption("Below are the pre-processed question and answer chunks returned by the RRF Rank Merger:")
            
            for idx, chunk in enumerate(pipeline_results["chunks"]):
                with st.container(border=True):
                    st.markdown(f"**Reference Chunk {idx+1}**")
                    
                    st.markdown(f"🎯 Cosine Similarity: `{chunk['cosine_similarity']:.4f}`")
                    
                    st.text_area("Chunk Text Content", chunk["text"], height=120, key=f"chunk_content_{idx}")
                    
        with tab_logs:
            st.markdown("### Pipeline Normalization Agent & Telemetry Logs")
            
            st.markdown(f"**Original User Query:** `{user_query}`")
            st.markdown(f"**Normalized Keywords (For BM25 & Qdrant Search):** `{norm_query}`")
            st.markdown(f"**Qdrant Collection Searched:** `{collection_name}`")
            st.markdown(f"**Vector Hits Fetched:** `{pipeline_results['vector_count']}`")
            st.markdown(f"**BM25 Hits Fetched:** `{pipeline_results['bm25_count']}`")
            
            st.divider()
            st.markdown("**System Prompt Sent to Vertex AI Mumbai:**")
            st.code(QA_SYSTEM_PROMPT, language="markdown")
