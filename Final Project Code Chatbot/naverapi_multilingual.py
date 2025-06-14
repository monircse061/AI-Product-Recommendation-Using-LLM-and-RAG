import gradio as gr
import requests
import random
import os
import numpy as np
from dataclasses import dataclass, field
from langdetect import detect
from openai import OpenAI

# -------------------- 🔧 Configuration --------------------
NAVER_CLIENT_ID = ""
NAVER_CLIENT_SECRET = ""
client = OpenAI(api_key="") 

# -------------------- 🌐 Language Utilities --------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # fallback

def translate(text, target_lang):
    lang_map = {
        "en": "English",
        "ko": "Korean",
        "ja": "Japanese",
        "th": "Thai"
    }

    target_name = lang_map.get(target_lang, "English")
    system_prompt = f"Translate the following text to {target_name}:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Translation Error]: {str(e)}"

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

# -------------------- 🔍 Search Logic with Translation --------------------
def search_products(input_text):
    input_lang = detect_language(input_text)
    
    # Translate user query to Korean
    query_for_naver = translate(input_text, "ko") if input_lang != "ko" else input_text
    products = fetch_naver_products(query_for_naver)
    
    output = ""
    for p in products:
        if "error" in p:
            output += f"❌ Error: {p['error']}<br/>{p['response']}<br/>\n"
        else:
            product_info_ko = (
                f"📦 상품명: {p['Title']}\n"
                f"💰 가격: {p['Price']} KRW\n"
                f"🌍 원산지: Korea\n"
                f"🔗 제품 보기: {p['Link']}"
            )
            translated_info = translate(product_info_ko, input_lang) if input_lang != "ko" else product_info_ko
            output += translated_info.replace("\n", "<br/>") + "<br/>"

            image_url = p.get("Image URL", "").strip()
            if image_url:
                output += f'<img src="{image_url}" width="300"/><br/>\n'
            else:
                output += "🖼️ <i>No image available</i><br/>\n"
            output += "<hr/>\n"
    return output

# -------------------- 🚀 Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🌐 Multilingual Naver Product Search (EN/KR/JA/TH)")
    query_input = gr.Textbox(label="Search Query", placeholder="e.g., ノートパソコン / โน้ตบุ๊ค / laptop / 노트북")
    search_btn = gr.Button("Search")
    output_box = gr.Markdown()

    search_btn.click(fn=search_products, inputs=query_input, outputs=output_box)

demo.launch(share=True, inbrowser=True)
