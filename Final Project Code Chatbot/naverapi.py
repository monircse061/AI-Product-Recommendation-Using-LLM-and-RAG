import gradio as gr
import requests
import random
import os
import numpy as np
from dataclasses import dataclass, field

# -------------------- 🔧 Configuration --------------------
NAVER_CLIENT_ID = ""
NAVER_CLIENT_SECRET = ""

# -------------------- 🛍️ Naver Shopping API --------------------
def fetch_naver_products(query, limit=3):
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
                "Product ID": item.get("productId", f"NAVER-{random.randint(10000,99999)}"),
                "Title": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                "Price": int(item.get("lprice", 0)),
                "Image URL": item.get("image", ""),
                "Link": item.get("link", "")
            })
    else:
        results.append({
            "error": f"API call failed with status {response.status_code}",
            "response": response.text
        })
    return results

def search_products(input_text):
    products = fetch_naver_products(input_text)
    output = ""
    for p in products:
        if "error" in p:
            output += f"❌ Error: {p['error']}<br/>{p['response']}<br/>\n"
        else:
            output += (
                f"📦 <b>상품명</b>: {p['Title']}<br/>\n"
                f"💰 <b>가격</b>: {p['Price']} KRW<br/>\n"
                f'<a href="{p["Link"]}" target="_blank">🔗 제품 보기</a><br/>\n'
            )
            image_url = p.get("Image URL", "").strip()
            if image_url:
                output += f'<img src="{image_url}" width="300"/><br/>\n'
            else:
                output += "🖼️ <i>No image available</i><br/>\n"
            output += "<hr/>\n"
    return output


# -------------------- 🚀 Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Naver Shopping Product Search")
    query_input = gr.Textbox(label="Search Query", placeholder="e.g., 노트북")
    search_btn = gr.Button("Search")
    output_box = gr.Markdown()
    

    search_btn.click(fn=search_products, inputs=query_input, outputs=output_box)

demo.launch(share=True, inbrowser=True)
