from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest

# -------------------- 1. Set up clients --------------------
client = OpenAI(api_key="")

qdrant_client = QdrantClient(
    url="", 
    api_key="",
)

collection_name = "products"

# -------------------- 2. Function to embed user query --------------------
def embed_query(client, query, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=query,
        model=model
    )
    return response.data[0].embedding

# -------------------- 3. Search in Qdrant --------------------
def search_products(client, user_query, top_k=5):
    # ✅ Fix: Pass client here
    query_vector = embed_query(client, user_query)

    # Perform vector search
    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    # Return payloads (metadata) from top results
    results = []
    for hit in hits:
        results.append({
            "score": hit.score,
            "상품코드": hit.payload.get("상품코드", ""),
            "상품명": hit.payload.get("원본상품명", ""),
            "가격": hit.payload.get("가격", ""),
            "배송비": hit.payload.get("배송비용", ""),
            "원산지": hit.payload.get("원산지", ""),
            "이미지": hit.payload.get("이미지소URL", ""),
        })
    return results

# -------------------- 4. Try a search --------------------
query = "학생 백팩"   # Example: "winter padded bag for women"
results = search_products(client, query)

for i, item in enumerate(results):
    print(f"\n🔹 Result {i+1}")
    print(f"상품코드: {item['상품코드']}")
    print(f"상품명: {item['상품명']}")
    print(f"가격: {item['가격']}원 / 배송비: {item['배송비']}원")
    print(f"원산지: {item['원산지']}")
    print(f"이미지 URL: {item['이미지']}")
    print(f"✅ Similarity Score: {item['score']:.4f}")