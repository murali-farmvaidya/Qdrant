import streamlit as st
import time
import pandas as pd
import rag_utils
import importlib
importlib.reload(rag_utils) # Force reload of the consolidated backend logic
from generate_excel import generate_excel_report
import io
import re

# Initialize clients
azure_client = rag_utils.init_azure_client()
qdrant = rag_utils.init_qdrant()
CHUNK_SIZES = rag_utils.CHUNK_SIZES

# Sync Tooling
if st.sidebar.button("🔄 Force Clear App Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache Cleared! Reloading...")
    st.rerun()

st.sidebar.caption("Last Sync: 2026-05-17 (Slumber Added)")

def render_single_result(label, size, result_data, q_idx):
    """Helper to render a single RAG result block with improved layout."""
    with st.container(border=True):
        st.markdown(f"#### 🏷️ {label}")
        
        # Answer Section
        st.markdown("**💬 Answer:**")
        st.info(result_data["answer"])
        
        # Metadata Section
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("📏 Chunk Size")
            st.write(f"`{size} tokens`")
        with c2:
            st.caption("🔍 Chunks Found")
            st.write(f"`{result_data['num_chunks']}`")
        with c3:
            st.caption("🎯 Precision@3")
            st.write(f"`{result_data['precision_at_3']:.4f}`")

        # Retrieved Chunks Section
        if result_data["chunks"]:
            with st.expander("📚 View Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(result_data["chunks"]):
                    st.markdown(f"**Chunk {i+1}** (Score: `{chunk['cosine_similarity']:.4f}`)")
                    st.text_area(f"Content {i+1}", chunk["text"], height=100, key=f"chunk_{q_idx}_{label}_{size}_{i}")
                    st.divider()
        else:
            st.warning("No chunks retrieved for this combination.")

# Streamlit UI
st.set_page_config(page_title="Agricultural Assistant", page_icon="🌾", layout="wide")

# Methodology Selection
methodology = st.sidebar.radio(
    "Chunking Strategy",
    ["Semantic", "Neural", "Recursive", "Slumber"],
    index=0,
    help="Slumber = Agentic LLM-driven chunking (GPT-4.1 decides exact split points)"
)

st.title(f"🌾 Agricultural Assistant - 10 Combinations ({methodology})")
st.caption(f"Compare answers across 5 different chunk sizes (128-2048) and 2 Embedding Models using strictly {methodology} Chunking on the rnf_A Knowledge Base")

with st.sidebar:
    st.header("⚙️ Architecture & Metadata")
    
    chunk_info = {
        "Semantic":  "Potion-32M (Fixed-Size Semantic)",
        "Neural":    "ModernBERT (Semantic Shift Detection)",
        "Recursive": "RecursiveCharacter (Standard Overlap)",
        "Slumber":   "GPT-4.1 Agentic Chunker (SlumberChunker)",
    }

    st.info(f"""
    - **Chunking:** {chunk_info[methodology]}
    - **Database:** Qdrant Cloud
    - **Search:** Cosine Similarity
    - **LLM:** GPT-4.1
    """)
    st.write("---")
    st.write("Total Combinations: **10**")

# Input Section
st.subheader("📤 Question Input")
uploaded_file = st.file_uploader("Upload a CSV file containing questions (Optional)", type=["csv", "txt"])
pasted_questions = st.text_area("OR Paste your questions below:", placeholder="Enter each question on a new line or separated by '?'", height=150)

top_k = st.slider("Number of references per chunk size", 1, 10, 3)

if st.button(f"🚀 Run 10-Combination {methodology} Analysis"):
    questions_list = []
    
    # Process uploaded file
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
            
    # Process pasted questions
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
        # Progress Tracking
        total_q = len(questions_list)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        started_at = time.perf_counter()

        for idx, q in enumerate(questions_list):
            status_text.text(f"Processing Question {idx+1}/{total_q}: {q[:50]}...")
            
            try:
                # 1. Retrieve chunks for all 10 combinations
                # OpenAI Embeddings
                chunks_openai, _ = rag_utils.retrieve_all_chunk_sizes(q, top_k, provider="openai", methodology=methodology.lower())
                # Google Embeddings
                chunks_google, _ = rag_utils.retrieve_all_chunk_sizes(q, top_k, provider="google", methodology=methodology.lower())
                
                # 2. Generate answers for all 10 combinations
                ans_openai = rag_utils.generate_answers_all_sizes(q, chunks_openai, azure_client)
                ans_google = rag_utils.generate_answers_all_sizes(q, chunks_google, azure_client)
                
                all_results.append({
                    "question": q,
                    "openai": ans_openai,
                    "google": ans_google
                })
                
                # Render results in UI immediately
                st.markdown(f"### ❓ Question {idx+1}")
                st.info(q)
                
                # Display results
                # Display results using Tabs for Models
                tab_openai, tab_google = st.tabs(["🌐 OpenAI (text-embedding-3-large)", "🔍 Google (gemini-embedding-2)"])
                
                with tab_openai:
                    st.markdown(f"### OpenAI text-embedding-3-large ({methodology})")
                    for size in CHUNK_SIZES:
                        render_single_result(f"OpenAI, {size} tokens", size, ans_openai[size], idx)
                
                with tab_google:
                    st.markdown(f"### Google gemini-embedding-2 ({methodology})")
                    for size in CHUNK_SIZES:
                        render_single_result(f"Google, {size} tokens", size, ans_google[size], idx)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing question '{q}': {e}")
            
            progress_bar.progress((idx + 1) / total_q)

        total_time = time.perf_counter() - started_at
        status_text.success(f"✅ Analysis Complete! Total time: {total_time:.2f}s")
        
        # Report Generation
        st.subheader("📊 Download Report")
        excel_data = generate_excel_report(all_results, CHUNK_SIZES)
        st.download_button(
            label="📥 Download Detailed Excel Report",
            data=excel_data,
            file_name=f"Agricultural_RAG_{methodology}_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
