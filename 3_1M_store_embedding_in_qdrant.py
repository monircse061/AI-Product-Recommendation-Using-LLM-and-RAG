import os
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# -------------------- Qdrant Setup --------------------
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "1M_products"

# Delete old collection if needed
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)
    print("🧹 Old collection deleted.")

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
print("Collection created.")

# -------------------- Chunk List --------------------
CHUNK_DIR = "embedded_chunks"
all_chunks = sorted([
    os.path.join(CHUNK_DIR, file)
    for file in os.listdir(CHUNK_DIR)
    if file.endswith(".pkl")
])
print(f"Total chunks found: {len(all_chunks)}")

# -------------------- Upload Log --------------------
UPLOAD_LOG = "uploaded_chunks.txt"
uploaded_chunks = set()
if os.path.exists(UPLOAD_LOG):
    with open(UPLOAD_LOG, "r") as f:
        uploaded_chunks = set(int(line.strip()) for line in f.readlines())

# -------------------- Upload Function --------------------
def store_chunk_to_qdrant(df, chunk_id, client, collection_name, batch_size=200):
    all_points = []
    for i, row in df.iterrows():
        embedding = row["embedding"]
        if embedding is not None:
            metadata = {
                "상품코드": str(row.get("상품코드", "")).strip(),
                "카테고리": str(row.get("카테고리명", "")).strip(),
                "원본상품명": str(row.get("원본상품명", "")).strip(),
                "모델": str(row.get("모델명", "")).strip(),
                "가격": int(row.get("오너클랜판매가", 0)),
                "배송비용": int(row.get("배송비", 0)),
                "원산지": str(row.get("원산지", "")).strip(),
                "이미지대URL": str(row.get("이미지대", "")).strip()
            }

            point = PointStruct(
                id=chunk_id * 10**6 + i,
                vector=embedding,
                payload=metadata
            )
            all_points.append(point)

    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    with open(UPLOAD_LOG, "a") as f:
        f.write(f"{chunk_id}\n")

# -------------------- Upload with Progress --------------------
remaining_chunks = [
    (idx, path) for idx, path in enumerate(all_chunks)
    if idx not in uploaded_chunks
]

for idx, chunk_path in tqdm(remaining_chunks, desc="🔄 Uploading Remaining Chunks", unit="chunk"):
    try:
        df_chunk = pd.read_pickle(chunk_path)
        store_chunk_to_qdrant(df_chunk, idx, qdrant_client, collection_name)
    except Exception as e:
        print(f"Error in chunk {idx}: {e}")
        break  # Or `continue` if you prefer to skip

print("All remaining chunks uploaded with progress bar.")
