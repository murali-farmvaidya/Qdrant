from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from config import QDRANT_CONFIGS, CHUNK_SIZES

def transfer_collection(client, old_name, new_name):
    try:
        # Check if old exists
        old_info = client.get_collection(old_name)
    except Exception as e:
        print(f"  [SKIP] {old_name} does not exist.")
        return

    print(f"  Transferring {old_info.points_count} points from {old_name} to {new_name}...")

    # Create new collection
    try:
        client.get_collection(new_name)
        print(f"    {new_name} already exists. Deleting it first...")
        client.delete_collection(new_name)
    except:
        pass
    
    # Get old vector size
    old_size = old_info.config.params.vectors.size
    
    client.create_collection(
        collection_name=new_name,
        vectors_config=VectorParams(size=old_size, distance=Distance.COSINE)
    )

    # Scroll and transfer points
    records, next_page = client.scroll(
        collection_name=old_name,
        limit=100,
        with_payload=True,
        with_vectors=True
    )
    
    total_transferred = 0
    while records:
        points = [
            PointStruct(id=r.id, vector=r.vector, payload=r.payload)
            for r in records
        ]
        client.upsert(collection_name=new_name, points=points)
        total_transferred += len(points)
        print(f"    Transferred {total_transferred}/{old_info.points_count} points...", end='\r')
        
        if next_page is None:
            break
            
        records, next_page = client.scroll(
            collection_name=old_name,
            limit=100,
            offset=next_page,
            with_payload=True,
            with_vectors=True
        )
    print(f"\n    ✓ Successfully transferred all points. Deleting {old_name}...")
    client.delete_collection(old_name)

def main():
    print("=========================================")
    print("Migrating angrau_ collections to rnf_google_")
    print("=========================================")

    # Method name to prefix mapping
    method_to_prefix = {
        "recursive": ("angrau_collection", "rnf_google_collection"),
        "semantic":  ("angrau_semantic_collection", "rnf_google_semantic_collection"),
        "neural":    ("angrau_neural_collection", "rnf_google_neural_collection"),
        "slumber":   ("angrau_slumber_collection", "rnf_google_slumber_collection")
    }

    for method, (old_prefix, new_prefix) in method_to_prefix.items():
        print(f"\nProcessing {method.upper()} cluster...")
        cfg = QDRANT_CONFIGS[method]
        client = QdrantClient(url=cfg["url"], api_key=cfg["api_key"], timeout=120)
        
        for size in CHUNK_SIZES:
            old_name = f"{old_prefix}_{size}"
            new_name = f"{new_prefix}_{size}"
            transfer_collection(client, old_name, new_name)

if __name__ == "__main__":
    main()
