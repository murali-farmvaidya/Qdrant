"""
process_and_upload_neural.py
================================
Ingests rnf_A.txt using NeuralChunker.
Creates 10 Qdrant collections:
  - rnf_A_neural_collection_{128,256,512,1024,2048}        -> OpenAI text-embedding-3-large
  - rnf_google_neural_collection_{128,256,512,1024,2048}   -> Google gemini-embedding-2
"""
import os
import time
import google.generativeai as genai
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from chonkie import RecursiveChunker

from config import (
    QDRANT_CONFIGS, GOOGLE_API_KEY, GOOGLE_EMBED_MODEL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_EMBED_DEPLOYMENT, CHUNK_SIZES, SOURCE_FILE
)

qdrant_cfg = QDRANT_CONFIGS["neural"]
qdrant = QdrantClient(url=qdrant_cfg["url"], api_key=qdrant_cfg["api_key"], timeout=120)
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
    try:
        if qdrant.get_collection(name).points_count > 0:
            print(f"  [SKIP] {name} already has points.", flush=True)
            return False
    except Exception: pass
    try: qdrant.delete_collection(name)
    except Exception: pass
    qdrant.create_collection(collection_name=name, vectors_config=VectorParams(size=size_val, distance=Distance.COSINE))
    return True

def upload(collection_name, chunks, embed_fn, batch_size=25):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        try:
            embeddings = embed_fn(batch)
            points = [PointStruct(id=i+j, vector=embeddings[j], payload={"text": batch[j]}) for j in range(len(batch))]
            qdrant.upsert(collection_name=collection_name, points=points)
            print(f"    [OK] {min(i+batch_size, len(chunks))}/{len(chunks)} uploaded", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"    [ERR] Batch {i} error: {e}", flush=True)
            time.sleep(5)

def get_neural_chunks(text, size):
    try:
        from chonkie import NeuralChunker
        chunker = NeuralChunker(chunk_size=size)
        return [c.text for c in chunker.chunk(text)]
    except Exception:
        pass
    try:
        from chonkie import SDPMChunker
        chunker = SDPMChunker(embedding_model="minishlab/potion-base-8M", chunk_size=size)
        return [c.text for c in chunker.chunk(text)]
    except Exception:
        pass
    chunker = RecursiveChunker(chunk_size=size)
    return [c.text for c in chunker.chunk(text)]

def main():
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    for size in CHUNK_SIZES:
        chunks = get_neural_chunks(text, size)
        if ensure_collection(f"rnf_A_neural_collection_{size}", size_val=3072):
            print(f"  [OpenAI] rnf_A_neural_collection_{size}")
            upload(f"rnf_A_neural_collection_{size}", chunks, embed_openai)
        if ensure_collection(f"rnf_google_neural_collection_{size}", size_val=3072):
            print(f"  [Google] rnf_google_neural_collection_{size}")
            upload(f"rnf_google_neural_collection_{size}", chunks, embed_google)

if __name__ == "__main__":
    main()
