import streamlit as st
import time
import pandas as pd
import io
import re
import sys
import os
import requests
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rag_utils
from config import CHUNK_SIZES
import importlib
importlib.reload(rag_utils)
from generate_excel import generate_excel_report

from google.oauth2 import service_account
from google.auth.transport.requests import Request
from bm25 import BM25Okapi

# Vertex AI Configuration
PROJECT_ID = os.getenv("VERTEX_SA_PROJECT_ID", "gen-lang-client-0043306559")
LOCATION = "asia-south1"
MODEL_ID = "gemini-2.5-flash"

@st.cache_resource
def get_vertex_token():
    private_key = os.getenv("VERTEX_SA_PRIVATE_KEY")
    if private_key:
        private_key = private_key.replace("\\n", "\n")
        
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

import json

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
            if not candidates:
                return f"No candidates returned. Raw data: {json.dumps(data)}"
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return f"No parts in content. Raw data: {json.dumps(data)}"
            return parts[0].get("text", "").strip()
        else:
            raise Exception(f"Vertex API returned {response.status_code}: {response.text}")
    except Exception as e:
        # Fallback seamlessly to Standard Google Generative AI Developer API using GOOGLE_API_KEY
        try:
            import google.generativeai as genai
            from config import GOOGLE_API_KEY
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

# ─── FLOWCHART STEP 2: Normalization Agent ──────────────────────────────────────
# The Normalization Agent intercepts the raw User Query and extracts entities/keywords
# using Vertex AI (Gemini 2.5 Flash) to maximize retrieval precision.
# Crucially, it must extract keywords in the SAME language as the query to enable
# precise token matching in local BM25 and multilingual vector indices.
def normalize_query(query):
    prompt = (
        "Extract the core agricultural entities and search keywords from this question. "
        "Return ONLY space-separated keywords. You MUST return them in the same language "
        "as the input question (e.g., if the question is in Telugu, you MUST return them in Telugu).\n"
        f"Question: {query}"
    )
    norm = call_vertex_gemini(prompt)
    return norm if norm and not "API Error" in norm else query

def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

@st.cache_data
def get_bm25_index(methodology, chunk_size):
    qdrant = rag_utils.init_qdrant(methodology)
    prefix_map = {
        "semantic":  "rnf_A_semantic_collection",
        "neural":    "rnf_A_neural_collection",
        "recursive": "rnf_A_collection",
        "slumber":   "rnf_A_slumber_collection",
    }
    collection_name = f"{prefix_map.get(methodology.lower(), 'rnf_A_collection')}_{chunk_size}"
    chunks = []
    try:
        # Fetching chunks directly from Qdrant to ensure exact match with Vector Search
        records, offset = qdrant.scroll(collection_name=collection_name, limit=2000, with_payload=True, with_vectors=False)
        for r in records:
            if r.payload and "text" in r.payload:
                chunks.append(r.payload["text"])
    except Exception as e:
        print(f"Error scrolling {collection_name}: {e}")
        
    if not chunks:
        # Fallback recursive chunker if Qdrant fetch fails
        source_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rnf_A.txt'))
        with open(source_file, "r", encoding="utf-8") as f:
            text = f.read()
        from chonkie import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=chunk_size)
        chunks = [c.text for c in chunker.chunk(text)]
        
    tokenized_chunks = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized_chunks), chunks

def process_combination(norm_query, raw_query, vector_chunks, methodology, size, top_k):
    # ─── FLOWCHART STEP 3 (A): BM25 Retrieval ─────────────────────────────────
    # Performs sparse keyword-based retrieval on the dynamically indexed Qdrant chunks
    bm25_index, raw_chunks = get_bm25_index(methodology, size)
    tokenized_query = norm_query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k*2]
    bm25_results = [{"text": raw_chunks[i], "score": bm25_scores[i]} for i in top_bm25_idx]
    
    # ─── FLOWCHART STEP 4: Merge Results (RRF) ──────────────────────────────────
    # Combines BM25 (Sparse) and Qdrant Vector (Dense) search results.
    # We apply Reciprocal Rank Fusion (RRF) to merge the ranked lists fairly.
    scores = {}
    for rank, hit in enumerate(bm25_results):
        t = hit["text"]
        scores[t] = scores.get(t, 0) + rrf_score(rank + 1)
    for rank, hit in enumerate(vector_chunks):
        t = hit["text"]
        scores[t] = scores.get(t, 0) + rrf_score(rank + 1)
        
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # ─── FLOWCHART STEP 5: Reranker (Top Chunks Selection) ────────────────────
    # Selects the final top K high-priority chunks based on the computed RRF scores.
    # To compute a fair, comparable semantic precision, we map the actual vector
    # cosine similarity from our dense retrieval hits (defaulting to minimum vector score if BM25-only).
    min_vector_score = min([c.get("cosine_similarity", 0.3) for c in vector_chunks]) if vector_chunks else 0.3
    vector_scores = {c["text"]: c.get("cosine_similarity", 0.3) for c in vector_chunks}
    
    top_merged = []
    for m in merged[:top_k]:
        text = m[0]
        rrf = m[1]
        cos_sim = vector_scores.get(text, min_vector_score)
        top_merged.append({
            "text": text,
            "cosine_similarity": cos_sim,
            "rrf_score": rrf
        })
    
    # ─── FLOWCHART STEP 6: Vertex AI Generation (Mumbai) ──────────────────────
    # Synthesizes the final response using the Top Chunks as context with Gemini 2.5 Flash on Vertex AI (Mumbai)
    context = "\n\n".join([f"Chunk {i+1}:\n{c['text']}" for i, c in enumerate(top_merged)])
    system_prompt = """You are a farming knowledge synthesis expert. Your task is to take knowledge chunks and synthesize them into concise, detailed answers in natural conversational Telugu.
ABSOLUTE RULES:
1. Keep answer to approximately 100 words
2. Include varieties, methods, fertilizer schedule
3. Telugu words for numbers (1.5 kg -> ఒకటిన్నర కిలోలు)
4. Natural conversational Telugu - NOT formal
5. No English words, no markdown headings, no bullet points
6. Synthesize chunks into one flowing answer
7. No greetings or preamble"""
    answer_prompt = f"Question: {raw_query}\n\nKnowledge Chunks:\n{context}"
    
    start_time = time.perf_counter()
    answer = call_vertex_gemini(answer_prompt, system_instruction=system_prompt)
    elapsed = time.perf_counter() - start_time
    
    # Calculate precision based on the actual vector cosine similarity of the top 3 merged chunks
    precision = sum(c["cosine_similarity"] for c in top_merged[:3]) / len(top_merged[:3]) if top_merged else 0
    return {
        "answer": answer,
        "chunks": top_merged,
        "precision_at_3": precision,
        "num_chunks": len(top_merged),
        "elapsed_time": elapsed
    }

def render_single_result(label, size, result_data, q_idx):
    with st.container(border=True):
        st.markdown(f"#### 🏷️ {label}")
        
        st.markdown("**💬 Answer:**")
        st.info(result_data["answer"])
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("📏 Chunk Size")
            st.write(f"`{size} tokens`")
        with c2:
            st.caption("🔍 Chunks Found")
            st.write(f"`{result_data['num_chunks']}`")
        with c3:
            st.caption("🎯 Precision@3 (Cosine Similarity)")
            st.write(f"`{result_data['precision_at_3']:.4f}`")
        with c4:
            st.caption("⏱️ Processing Time")
            st.write(f"`{result_data.get('elapsed_time', 0.0):.2f}s`")

        if result_data["chunks"]:
            with st.expander("📚 View Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(result_data["chunks"]):
                    st.markdown(f"**Chunk {i+1}** | Cosine Similarity: `{chunk['cosine_similarity']:.4f}` | RRF Score: `{chunk['rrf_score']:.4f}`")
                    st.text_area(f"Content {i+1}", chunk["text"], height=100, key=f"chunk_{q_idx}_{label}_{size}_{i}")
                    st.divider()
        else:
            st.warning("No chunks retrieved for this combination.")

# ─── STREAMLIT UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Flowchart Retriever Comparison", page_icon="🧩", layout="wide")

# Initialize session state for multi-select
if "selected_meths" not in st.session_state:
    st.session_state.selected_meths = ["Semantic", "Neural"]

# Add buttons to Select All or Clear All strategies in the sidebar
st.sidebar.markdown("### 🎛️ Strategy Quick Select")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("Select All", use_container_width=True):
        st.session_state.selected_meths = ["Semantic", "Neural", "Recursive", "Slumber"]
with col2:
    if st.sidebar.button("Clear All", use_container_width=True):
        st.session_state.selected_meths = []

# Multiselect linked to session state key
selected_methodologies = st.sidebar.multiselect(
    "Chunking Strategy",
    ["Semantic", "Neural", "Recursive", "Slumber"],
    key="selected_meths",
    help="Select one or more chunking methodologies. The flowchart pipeline will process and compare all of them at once."
)

if not selected_methodologies:
    st.title("🧩 Flowchart Pipeline - Multi-Strategy Comparison")
    st.warning("Please select at least one Chunking Strategy from the sidebar to proceed.")
else:
    meth_desc = " + ".join(selected_methodologies)
    st.title(f"🧩 Flowchart Pipeline - Multi-Strategy Comparison ({meth_desc})")
    st.caption(f"Process: User Query -> Normalization -> BM25 + Vector -> Merge (RRF) -> Top Chunks -> Vertex Generation")

    with st.sidebar:
        st.header("⚙️ Architecture & Metadata")
        
        chunk_info = {
            "Semantic":  "Potion-32M (Fixed Semantic)",
            "Neural":    "ModernBERT (Shift Detection)",
            "Recursive": "RecursiveCharacter (Overlap)",
            "Slumber":   "GPT-4.1 Agentic (Slumber)",
        }

        # Render a perfectly aligned Markdown Table for active strategies
        st.markdown("**Active Strategy Mapping:**")
        meta_rows = [f"| **{m}** | {chunk_info[m]} |" for m in selected_methodologies]
        st.markdown(f"""
| Strategy | Engine Configuration |
| :--- | :--- |
{"\n".join(meta_rows)}
""")
        
        st.info("""
        **Benchmark Specs:**
        - **Vector DB:** Qdrant Cloud
        - **Sparse DB:** Local BM25
        - **Merge:** Reciprocal Rank Fusion
        - **LLM:** Vertex 2.5 Flash (Mumbai)
        """)
        st.write("---")
        total_combos = len(selected_methodologies) * 10
        st.write(f"Total Active Combinations: **{total_combos}**")

    st.subheader("📤 Question Input")
    uploaded_file = st.file_uploader("Upload a CSV file containing questions (Optional)", type=["csv", "txt"])
    pasted_questions = st.text_area("OR Paste your questions below:", placeholder="Enter each question on a new line or separated by '?'", height=150)
    top_k = st.slider("Number of references per chunk size", 1, 10, 3)

    if st.button(f"🚀 Run {total_combos}-Combination Flowchart Analysis"):
        questions_list = []
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(io.StringIO(content))
                    col_name = "question" if "question" in df.columns else df.columns[0]
                    questions_list.extend(df[col_name].dropna().tolist())
                else:
                    questions_list.extend([q.strip() for q in content.split("\n") if q.strip()])
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
        if pasted_questions.strip():
            splits = re.split(r'\?\s*|\n', pasted_questions)
            for s in splits:
                s = s.strip()
                if s:
                    if not s.endswith('?'): s += '?'
                    if s not in questions_list:
                        questions_list.append(s)

        if not questions_list:
            st.warning("Please provide at least one question.")
        else:
            total_q = len(questions_list)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a highly visible, persistent processing container
            live_status_container = st.empty()
            
            all_results = []
            started_at = time.perf_counter()

            for idx, q in enumerate(questions_list):
                status_text.text(f"Processing Question {idx+1}/{total_q}: {q[:50]}...")
                
                try:
                    # ─── FLOWCHART STEP 1: User Query captures the incoming raw input ─────
                    
                    # ─── FLOWCHART STEP 2: Normalization Agent processes the raw input ────
                    with live_status_container.container(border=True):
                        st.markdown("### 🏃 Current Pipeline Trace")
                        st.info(f"**Question:** {q}")
                        st.warning("🔄 **Step:** Running Query Normalization Agent...")
                    
                    norm_query = normalize_query(q)
                    
                    # Store results structured by selected methodologies
                    q_res = {
                        "question": q,
                        "normalized_query": norm_query,
                        "methodologies": {}
                    }
                    
                    st.markdown(f"### ❓ Question {idx+1}")
                    st.info(q)
                    st.caption(f"Normalized Context: {norm_query}")
                    
                    for meth in selected_methodologies:
                        # ─── FLOWCHART STEP 3 (B): Vector Retrieval ───────────────────────────
                        with live_status_container.container(border=True):
                            st.markdown("### 🏃 Current Pipeline Trace")
                            st.info(f"**Question:** {q}")
                            st.success(f"**Active Strategy:** {meth}")
                            st.warning("🔍 **Step:** Performing hybrid BM25 + Qdrant Cloud Vector Retrieval...")
                            
                        chunks_openai, _ = rag_utils.retrieve_all_chunk_sizes(norm_query, top_k*2, provider="openai", methodology=meth.lower())
                        chunks_google, _ = rag_utils.retrieve_all_chunk_sizes(norm_query, top_k*2, provider="google", methodology=meth.lower())
                        
                        ans_openai = {}
                        ans_google = {}
                        
                        for size in CHUNK_SIZES:
                            with live_status_container.container(border=True):
                                st.markdown("### 🏃 Current Pipeline Trace")
                                st.info(f"**Question:** {q}")
                                st.success(f"**Active Strategy:** {meth} (Size: {size} tokens)")
                                st.warning("🧬 **Step:** Merging ranks (RRF) & generating answer via Vertex AI (Mumbai)...")
                                
                            with st.spinner(f"Processing OpenAI ({meth}, {size} tokens)..."):
                                ans_openai[size] = process_combination(
                                    norm_query, q, chunks_openai.get(size, []), meth, size, top_k
                                )
                            with st.spinner(f"Processing Google ({meth}, {size} tokens)..."):
                                ans_google[size] = process_combination(
                                    norm_query, q, chunks_google.get(size, []), meth, size, top_k
                                )
                            
                        q_res["methodologies"][meth] = {
                            "openai": ans_openai,
                            "google": ans_google
                        }
                        
                        # Display Results beautifully for this strategy
                        with st.expander(f"🌾 {meth} Chunking Strategy - 10 Combinations", expanded=True):
                            tab_openai, tab_google = st.tabs(["🌐 OpenAI (text-embedding-3-large) Vectors", "🔍 Google (gemini-embedding-2) Vectors"])
                            
                            with tab_openai:
                                st.markdown(f"### OpenAI Vectors + BM25 Fusion ({meth}) -> Vertex AI")
                                for size in CHUNK_SIZES:
                                    render_single_result(f"OpenAI, {size} tokens", size, ans_openai[size], f"{idx}_{meth}")
                            
                            with tab_google:
                                st.markdown(f"### Google Vectors + BM25 Fusion ({meth}) -> Vertex AI")
                                for size in CHUNK_SIZES:
                                    render_single_result(f"Google, {size} tokens", size, ans_google[size], f"{idx}_{meth}")
                    
                    all_results.append(q_res)
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error processing question '{q}': {e}")
                
                progress_bar.progress((idx + 1) / total_q)

            # Clear live status container when done
            live_status_container.empty()
            
            total_time = time.perf_counter() - started_at
            status_text.success(f"✅ Analysis Complete! Total time: {total_time:.2f}s")
            
            st.subheader("📊 Download Report")
            excel_data = generate_excel_report(all_results, CHUNK_SIZES)
            st.download_button(
                label="📥 Download Side-by-Side Excel Report (All Strategies)",
                data=excel_data,
                file_name="Chunk_Retriever_Side_by_Side_Comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
