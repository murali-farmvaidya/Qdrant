# 🌾 FarmVaidya Hybrid RAG Pipeline

This repository contains the advanced **Hybrid Retrieval-Augmented Generation (RAG) Pipeline** for the FarmVaidya agricultural chatbot. The system is designed to provide Indian farmers with highly accurate, localized, and context-aware answers in natural conversational Telugu.

It leverages a dual-retrieval system combining **Qdrant Vector Database** (Dense Retrieval) and **BM25** (Sparse Retrieval), mathematically merged using **Reciprocal Rank Fusion (RRF)**, and powered by **Google Vertex AI (Gemini 2.5 Flash)**.

---

## 🏗️ Master Pipeline Architecture

The core pipeline (`app_qa.py`) executes a strict 6-step lifecycle to guarantee accurate and concise answers:

1. **User Query Input:** Accepts raw Telugu questions from the user.
2. **Normalization Agent:** Uses Gemini 2.5 Flash to strip conversational fluff and extract pure agricultural keywords.
3. **Dual Retrieval:**
   - **Dense Vector Search:** Embeds the query and fetches semantically similar chunks from Qdrant Cloud.
   - **Sparse Keyword Search:** Runs a local BM25 search to ensure critical entity names (crops, chemicals, pests) are matched exactly.
4. **Reciprocal Rank Fusion (RRF):** Merges the results of both retrieval methods mathematically (`Score = 1.0 / (60 + rank)`).
5. **Reranker:** Truncates the merged list to the absolute Top K best reference chunks.
6. **Synthesis Agent:** Vertex AI (Gemini 2.5 Flash) generates a final response restricted by strict absolute rules (approx. 100 words, conversational Telugu, no English words).

---

## 📂 Repository Structure

- `chunk_retriever/`
  - `app_qa.py` - The main Streamlit UI testing the full 6-Step Hybrid RAG pipeline.
  - `app.py` - Standard chunk retriever UI for model and chunk strategy comparison.
  - `bm25.py` - Local implementation of the BM25 sparse retrieval logic.
- `process_and_upload_*.py` - Scripts responsible for parsing text files, chunking data (Recursive, Semantic, Neural), generating embeddings, and batch-uploading `PointStructs` to Qdrant Cloud.
- `config.py` - Central configuration module for mapping embedding models, LLM deployments, and Qdrant collections. *(Note: Secrets are safely omitted and rely on `.env`)*
- `rag_utils.py` - Shared utilities for generating embeddings via Azure OpenAI and Google APIs.
- `generate_excel.py` - Generates benchmarking reports dynamically.

---

## 🚀 Getting Started

### 1. Environment Setup

Create a `.env` file in the root directory (this file is git-ignored) and populate it with your credentials:

```env
# Google & Vertex AI
GOOGLE_API_KEY=your_google_api_key
VERTEX_SA_PROJECT_ID=vertex-ai-project-494805

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Qdrant Cloud Collections
QA_QDRANT_URL=your_qdrant_url
QA_QDRANT_API_KEY=your_qdrant_api_key
# (Add keys for RECURSIVE, SEMANTIC, NEURAL as needed)
```

If using Vertex AI via Service Account, ensure your `vertex_sa.json` file is securely placed in the root directory.

### 2. Ingesting Knowledge into Qdrant

If you need to update the Qdrant Cloud collections, run the ingestion scripts. For example, to upload the Telugu Q&A dataset:

```bash
python process_and_upload_qa.py
```

### 3. Running the Streamlit UI

To launch the Hybrid RAG testing dashboard:

```bash
cd chunk_retriever
streamlit run app_qa.py
```

---

## 📊 Chunking Strategies Benchmarked

The codebase contains infrastructure to benchmark multiple chunking methodologies against each other:
- **Recursive Chunking**
- **Semantic Chunking** (sentence-transformer based)
- **Neural Chunking** (BERT-based semantic shift detection)
- **Pre-processed Q&A Pairs** (rnf_qa.txt)

---

## 🔒 Security

This repository utilizes strict `.gitignore` rules to prevent the accidental leakage of sensitive information. Ensure you **never** commit `.env`, `*.pem`, `*.key`, or `*sa.json` files.
