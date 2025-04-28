import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
import random
import requests

# -------------------- Configuration --------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "1M_products"
VECTOR_DIM = 1536
OPENAI_API_KEY = ""  # Replace with your key
NAVER_CLIENT_ID = ""
NAVER_CLIENT_SECRET = ""

# -------------------- Initialize Clients --------------------
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------- Embedding Function --------------------
def embed_text(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

# -------------------- Qdrant Search --------------------
def search_qdrant(query, top_k=5):
    query_vector = embed_text(query)
    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return [hit.payload for hit in hits]

# -------------------- Naver API Call --------------------
def fetch_naver_products(query, limit=2):
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": limit
    }
    response = requests.get(url, headers=headers, params=params)
    results = []

    if response.status_code == 200:
        items = response.json().get("items", [])
        for item in items:
            results.append({
                "상품코드": item.get("productId", f"NAVER-{random.randint(10000,99999)}"),
                "원본상품명": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                "가격": int(item.get("lprice", 0)),
                "배송비용": 2500,
                "원산지": "한국",
                "이미지대URL": item.get("image", "https://via.placeholder.com/150")
            })
    return results

# -------------------- LLM Reasoning --------------------
def generate_response_with_llm(query, product_infos, history=[]):
    context = "\n".join([f"- {p['원본상품명']} ({p['가격']}원, {p['원산지']})" for p in product_infos])
    prompt = (
        f"You are a helpful shopping assistant. A user asked: '{query}'.\n\n"
        f"Here are 5 products:\n{context}\n\n"
        "Give a friendly product recommendation summary."
    )
    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content

# -------------------- Chatbot Logic --------------------
def chatbot_response(user_input, history):
    history = history[-5:] if history else []

    combined_query = " ".join([msg["content"] for msg in history if msg["role"] == "user"])
    combined_query += " " + user_input

    ownerclan_results = search_qdrant(combined_query, top_k=5)
    ownerclan_top3 = ownerclan_results[:3]
    naver_results = fetch_naver_products(user_input, limit=2)
    all_products = ownerclan_top3 + naver_results

    reasoning = generate_response_with_llm(user_input, all_products, history)

    product_details = ""
    for p in all_products:
        product_details += (
            f"**🆔 상품코드**: {p['상품코드']}\n"
            f"**📦 상품명**: {p['원본상품명']}\n"
            f"**💰 가격**: {p['가격']}원\n"
            f"**🚚 배송비용**: {p['배송비용']}원\n"
            f"**🌍 원산지**: {p['원산지']}\n"
        )
        image_url = p.get("이미지대URL", "").strip()
        if image_url and image_url.startswith("http"):
            product_details += f"![image]({image_url})\n\n"
        else:
            product_details += "🖼️ *(No image available)*\n\n"
        product_details += "---\n"

    if "no" in user_input.lower():
        product_details += random.choice([
            "🔍 Do you prefer a specific brand or color?",
            "🎨 Would you like to explore a different style or category?",
            "🏷️ Are you looking for products in a different price range?"
        ])

    full_response = reasoning + "\n\n" + product_details

    return {"role": "assistant", "content": full_response}, history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": full_response}
    ]

# -------------------- Gradio UI --------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🛍️ Smart Picks: Your AI-Powered Shopping Buddy 🤖")
    gr.Markdown(
        """
        🎉 **Welcome to the AI-Powered Product Recommendation Assistant!** 🛍️🤖🧠  
        Ask anything like:  
        - 👜 "여름용 여성 백팩 추천해줘"  
        - 💼 "Affordable winter bags"  
        👉 You’ll get **TOP 5 products** with details and 🖼️ image previews!
        """
    )

    chatbot = gr.Chatbot(label="Smart Product Assistant", type="messages", height=520)
    state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(label="Ask about products", placeholder="e.g., 여성 여름 원피스 추천해줘")
        submit_btn = gr.Button("🔍 Ask")

    clear_btn = gr.Button("🧹 Clear Chat")

    def handle_input(user_input, history):
        response, new_history = chatbot_response(user_input, history)
        return "", new_history, new_history

    submit_btn.click(handle_input, [user_input, state], [user_input, chatbot, state])
    user_input.submit(handle_input, [user_input, state], [user_input, chatbot, state])
    clear_btn.click(lambda: [], None, chatbot)

demo.launch()
 # cd qdrant
 # cargo run --release