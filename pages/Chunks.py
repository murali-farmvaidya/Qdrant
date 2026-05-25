import time

import streamlit as st

from rag_utils import retrieve_cloud_chunks

st.set_page_config(page_title="Chunks Viewer", page_icon="🧩", layout="centered")

st.title("🧩 Chunks Viewer")
st.caption("Use this page to inspect the chunks retrieved from cloud storage.")

with st.sidebar:
    st.header("⚙️ Architecture \u0026 Metadata")
    st.markdown("""
    **1. Document Processing**
    - **Engine:** chonkie (RecursiveChunker)
    - **Size:** 1024 tokens per chunk

    **2. Embedding Model**
    - **Provider:** Azure OpenAI
    - **Model:** text-embedding-3-large
    - **Dimensionality:** 3072 dimensions
    - **Language:** Multilingual (Telugu semantics)

    **3. Vector Database**
    - **Engine:** Qdrant
    - **Deployment:** Cloud Storage (Collection: `rnf_A_collection`)

    **4. Search Algorithm**
    - **Type:** Dense Vector Semantic Search (HNSW)
    - **Distance Metric:** Cosine Similarity

    **5. Reranking / Pipeline**
    - **Current:** Explicit Cosine Reranking
    - **LLM:** gpt-4o (Azure OpenAI)
    """)

with st.form("chunks_form"):
    query = st.text_area(
        "Question",
        placeholder="ఉదాహరణ: అంతరపంటలు అంటే ఏమిటి? ఎందుకు అవసరం?",
        height=100,
    )
    top_k = st.slider("Number of chunks to show", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("Show chunks")

if submitted and query.strip():
    started_at = time.perf_counter()
    with st.spinner("Retrieving chunks..."):
        try:
            chunks, _ = retrieve_cloud_chunks(query, top_k)

            elapsed_seconds = time.perf_counter() - started_at
            st.metric("Response latency", f"{elapsed_seconds:.2f} seconds")
            st.markdown("**Question**")
            st.write(query)
            st.markdown("**Retrieved Chunks**")

            for index, chunk in enumerate(chunks, start=1):
                with st.container(border=True):
                    st.markdown(f"**Chunk {index}**")
                    st.caption(
                        f"Qdrant score: {chunk['qdrant_score']:.4f}"
                    )
                    st.write(chunk["text"])
        except Exception as exc:
            st.error(f"Error while retrieving chunks: {exc}")
elif submitted:
    st.warning("Please type a question first.")
