"""
config.py — Central configuration.
Loads credentials from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from current directory

# ── SHARED: Google Gemini Embedding ───────────────────────────────────────────
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL","models/gemini-embedding-2")

# ── SHARED: Azure OpenAI ──────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "https://agent-api-key2.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_EMBED_DEPLOYMENT  = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-large")
AZURE_CHAT_DEPLOYMENT   = os.getenv("AZURE_CHAT_DEPLOYMENT",  "gpt-4.1")

# ── Knowledge Base ────────────────────────────────────────────────────────────
SOURCE_FILE  = os.getenv("SOURCE_FILE", "rnf_A.txt")
CHUNK_SIZES  = [128, 256, 512, 1024, 2048]

# ── QDRANT CLUSTERS ───────────────────────────────────────────────────────────
# Each chunking technique has its own Qdrant Cluster API keys
QDRANT_CONFIGS = {
    "recursive": {
        "url": os.getenv("RECURSIVE_QDRANT_URL", "https://8fbdd3fe-6642-4c9e-a7ef-801bfc83fc8d.us-east-1-1.aws.cloud.qdrant.io:6333"),
        "api_key": os.getenv("RECURSIVE_QDRANT_API_KEY")
    },
    "semantic": {
        "url": os.getenv("SEMANTIC_QDRANT_URL", "https://8fbdd3fe-6642-4c9e-a7ef-801bfc83fc8d.us-east-1-1.aws.cloud.qdrant.io:6333"),
        "api_key": os.getenv("SEMANTIC_QDRANT_API_KEY")
    },
    "neural": {
        "url": os.getenv("NEURAL_QDRANT_URL", "https://8fbdd3fe-6642-4c9e-a7ef-801bfc83fc8d.us-east-1-1.aws.cloud.qdrant.io:6333"),
        "api_key": os.getenv("NEURAL_QDRANT_API_KEY")
    },
    "slumber": {
        "url": os.getenv("SLUMBER_QDRANT_URL", "https://8fbdd3fe-6642-4c9e-a7ef-801bfc83fc8d.us-east-1-1.aws.cloud.qdrant.io:6333"),
        "api_key": os.getenv("SLUMBER_QDRANT_API_KEY")
    },
    "qa": {
        "url": os.getenv("QA_QDRANT_URL", "https://8fbdd3fe-6642-4c9e-a7ef-801bfc83fc8d.us-east-1-1.aws.cloud.qdrant.io:6333"),
        "api_key": os.getenv("QA_QDRANT_API_KEY")
    }
}
