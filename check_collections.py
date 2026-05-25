from qdrant_client import QdrantClient
from config import QDRANT_CONFIGS

print("=========================================")
print("Checking all Qdrant Clusters")
print("=========================================")

for method, cfg in QDRANT_CONFIGS.items():
    print(f"\nCluster: {method.upper()}")
    print("-" * 50)
    try:
        client = QdrantClient(url=cfg["url"], api_key=cfg["api_key"], timeout=10)
        collections = client.get_collections().collections
        if not collections:
            print("  No collections found.")
        for collection in collections:
            info = client.get_collection(collection.name)
            print(f"  {collection.name:<40} {info.points_count:>6} points")
    except Exception as e:
        print(f"  Connection error: {e}")
