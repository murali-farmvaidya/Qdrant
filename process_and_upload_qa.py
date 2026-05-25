"""
process_and_upload_qa.py
========================
Ingests rnf_q_a.txt containing pre-chunked Q&A pairs (63 chunks).
Creates 2 Qdrant collections:
  - rnf_qa_collection_512         -> OpenAI text-embedding-3-large (3072 dim)
  - rnf_qa_google_collection_512  -> Google gemini-embedding-2 (3072 dim)

Uploads the entire unmodified chunk as a single text payload block.
"""

import os
import re
import time
import google.generativeai as genai
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from config import (
    QDRANT_CONFIGS, GOOGLE_API_KEY, GOOGLE_EMBED_MODEL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_EMBED_DEPLOYMENT
)

# Source Q&A text file
QA_SOURCE_FILE = "rnf_q_a.txt"

# Select the new QA Qdrant Cluster settings
qdrant_cfg = QDRANT_CONFIGS["qa"]
print("Initializing Qdrant client with Cloud endpoint:", qdrant_cfg["url"])
qdrant = QdrantClient(url=qdrant_cfg["url"], api_key=qdrant_cfg["api_key"], timeout=120)

# Configure embedding models
genai.configure(api_key=GOOGLE_API_KEY)
azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

def embed_openai(texts):
    resp = azure_client.embeddings.create(model=AZURE_EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]

def embed_google(texts):
    result = genai.embed_content(model=GOOGLE_EMBED_MODEL, content=texts, task_type="retrieval_document")
    return result.get("embedding", result.get("embeddings"))

def ensure_collection(name, size_val=3072):
    print(f"Ensuring collection: '{name}' exists (Dimension: {size_val})...")
    try:
        # Check if collection exists and has points already
        coll_info = qdrant.get_collection(name)
        if coll_info.points_count > 0:
            print(f"  [SKIP] Collection '{name}' already exists and contains {coll_info.points_count} points.")
            return False
    except Exception:
        # Collection does not exist, we will create it
        pass

    try:
        qdrant.delete_collection(name)
    except Exception:
        pass

    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size_val, distance=Distance.COSINE)
    )
    print(f"  [CREATED] Fresh collection '{name}' created successfully.")
    return True

def upload(collection_name, chunks, embed_fn, provider_name, batch_size=10):
    print(f"Starting upload of {len(chunks)} chunks to '{collection_name}' ({provider_name})...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            embeddings = embed_fn(batch)
            points = []
            for j in range(len(batch)):
                idx = i + j
                points.append(
                    PointStruct(
                        id=idx,
                        vector=embeddings[j],
                        payload={"text": batch[j]}
                    )
                )
            
            qdrant.upsert(collection_name=collection_name, points=points)
            print(f"  [OK] Batch {i//batch_size + 1}: Chunks {i+1} to {min(i+batch_size, len(chunks))} uploaded successfully.", flush=True)
            time.sleep(1.5)  # Rate limit safety sleep
        except Exception as e:
            print(f"  [ERROR] Failed to upload batch starting at index {i}: {e}", flush=True)
            time.sleep(5)

def parse_qa_chunks(file_path):
    print(f"Parsing Q&A chunks from file: '{file_path}'...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file '{file_path}' not found!")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split content by "Chunk X" headers (case-insensitive, multiline)
    raw_chunks = re.split(r"^Chunk \d+", content, flags=re.MULTILINE | re.IGNORECASE)
    
    chunks = []
    for rc in raw_chunks:
        rc = rc.strip()
        if not rc:
            continue
        
        # Keep physical line breaks inside the chunk to preserve question vs answer division
        lines = [line.strip() for line in rc.split("\n") if line.strip()]
        if not lines:
            continue
        
        full_text = "\n".join(lines)
        chunks.append(full_text)
        
    print(f"Successfully parsed {len(chunks)} pre-chunked Q&A pairs.")
    return chunks

def main():
    print("=" * 60)
    print("STARTING Q&A DATASET INGESTION & UPLOAD")
    print("=" * 60)
    
    # 1. Parse chunks
    chunks = parse_qa_chunks(QA_SOURCE_FILE)
    
    # 2. Upload to OpenAI Collection
    openai_col = "rnf_qa_collection_512"
    if ensure_collection(openai_col, size_val=3072):
        upload(openai_col, chunks, embed_openai, "OpenAI")
        
    # 3. Upload to Google Collection
    google_col = "rnf_qa_google_collection_512"
    if ensure_collection(google_col, size_val=3072):
        upload(google_col, chunks, embed_google, "Google")

    print("\n" + "=" * 60)
    print("ALL CHUNKS AND EMBEDDINGS SUCCESSFULLY UPLOADED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
