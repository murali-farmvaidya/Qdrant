"""
process_and_upload_slumber.py
================================
Ingests rnf_A.txt using SlumberChunker.
Creates 10 Qdrant collections:
  - rnf_A_slumber_collection_{128,256,512,1024,2048}        -> OpenAI text-embedding-3-large
  - rnf_google_slumber_collection_{128,256,512,1024,2048}   -> Google gemini-embedding-2

Optimizations:
1. Parallel Block Parsing: Uses ThreadPoolExecutor to run LLM boundary detection in parallel.
2. JSON Caching: Saves chunk results to disk so subsequent runs or restarts take 0 seconds.
3. Explicit Sizing (3072): Correctly matches OpenAI and Google Gemini 2 dimensions.
"""
import os
import time
import json
import google.generativeai as genai
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from chonkie import SlumberChunker
from chonkie.genie import AzureOpenAIGenie
from concurrent.futures import ThreadPoolExecutor

from config import (
    QDRANT_CONFIGS, GOOGLE_API_KEY, GOOGLE_EMBED_MODEL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_EMBED_DEPLOYMENT, AZURE_CHAT_DEPLOYMENT, CHUNK_SIZES, SOURCE_FILE
)

BLOCK_SIZE = 50_000
CACHE_DIR = "slumber_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

qdrant_cfg = QDRANT_CONFIGS["slumber"]
qdrant = QdrantClient(url=qdrant_cfg["url"], api_key=qdrant_cfg["api_key"], timeout=120)
genai.configure(api_key=GOOGLE_API_KEY)
azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# AzureOpenAI Genie - GPT-4.1 decides where to split each block
slumber_genie = AzureOpenAIGenie(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_api_key=AZURE_OPENAI_API_KEY,
    deployment=AZURE_CHAT_DEPLOYMENT,
    model=AZURE_CHAT_DEPLOYMENT,
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

def process_single_block(block_info):
    bi, block, size = block_info
    t0 = time.perf_counter()
    chunker = SlumberChunker(
        genie=slumber_genie,
        chunk_size=size,
        candidate_size=max(32, size // 4),
        verbose=False
    )
    try:
        objs = chunker.chunk(block)
        chunks = [c.text for c in objs]
        print(f"  Block {bi} -> {len(chunks)} chunks in {time.perf_counter()-t0:.1f}s", flush=True)
        return bi, chunks
    except Exception as e:
        print(f"  [ERR] Block {bi} failed: {e}", flush=True)
        return bi, []

def slumber_chunk_text(text, size):
    cache_path = os.path.join(CACHE_DIR, f"slumber_chunks_{size}.json")
    if os.path.exists(cache_path):
        print(f"  [CACHE LOAD] Loaded chunking from cache for size={size}!", flush=True)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    blocks = [text[i: i + BLOCK_SIZE] for i in range(0, len(text), BLOCK_SIZE)]
    print(f"  Chunking {len(blocks)} blocks in parallel (max 4 threads)...", flush=True)
    
    block_infos = [(bi, block, size) for bi, block in enumerate(blocks, 1)]
    
    all_chunks_dict = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_single_block, block_infos)
        for bi, chunks in results:
            all_chunks_dict[bi] = chunks
            
    # Assemble in order
    all_chunks = []
    for bi in sorted(all_chunks_dict.keys()):
        all_chunks.extend(all_chunks_dict[bi])
        
    # Cache to disk
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
    return all_chunks

def main():
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} chars from {SOURCE_FILE}")
    print(f"Slumber chunking running with Parallel Acceleration.")

    for size in CHUNK_SIZES:
        oai_name  = f"rnf_A_slumber_collection_{size}"
        goog_name = f"rnf_google_slumber_collection_{size}"
        
        try: oai_done = qdrant.get_collection(oai_name).points_count > 0
        except: oai_done = False
        try: goog_done = qdrant.get_collection(goog_name).points_count > 0
        except: goog_done = False
        
        if oai_done and goog_done:
            print(f"  [SKIP ALL] Size={size} already populated.", flush=True)
            continue

        print(f"\n{'='*55}")
        print(f"  SLUMBER - chunk_size={size}")
        print(f"{'='*55}")

        chunks = slumber_chunk_text(text, size)
        print(f"  Total chunks: {len(chunks)}")
        
        if ensure_collection(oai_name, size_val=3072):
            print(f"  [OpenAI] {oai_name}")
            upload(oai_name, chunks, embed_openai)
            
        if ensure_collection(goog_name, size_val=3072):
            print(f"  [Google] {goog_name}")
            upload(goog_name, chunks, embed_google)

    print("\n  Slumber ingestion complete!")

if __name__ == "__main__":
    main()
