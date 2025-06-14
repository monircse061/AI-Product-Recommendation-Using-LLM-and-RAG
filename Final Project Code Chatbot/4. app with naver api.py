import gradio as gr
from openai import OpenAI
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
import random
import os
from dataclasses import dataclass, field
import numpy as np
import io
import tempfile
import requests
from pydub import AudioSegment
import subprocess
from langdetect import detect
import sys

# -------------------- 🔧 Configuration --------------------
NAVER_CLIENT_ID = ""
NAVER_CLIENT_SECRET = ""
OPENAI_API_KEY = ""
QDRANT_URL = ""
QDRANT_API_KEY = ""
COLLECTION_NAME = "products"
VECTOR_DIM = 1536

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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
        response = openai_client.chat.completions.create(
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

#-------------------AppState and voice processing function------------------------------
@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    stopped: bool = False
    started_talking: bool = True
    conversation: list = field(default_factory=list)

def determine_pause(stream, sampling_rate, state):
    # Dummy pause detection — replace with your actual logic
    return len(stream) / sampling_rate > 2  # Treat as pause if >2 seconds

def process_audio(audio: tuple, state: AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream = np.concatenate((state.stream, audio[1]))
    state.pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    if state.pause_detected and state.started_talking:
        return gr.Audio(recording=False), state
    return None, state

def speaking(audio_bytes):  # Dummy streaming generator
    yield audio_bytes

#------------------------------Response handler for voice-------------------------
def response(state: AppState):
    if not state.pause_detected and not state.started_talking:
        return None, AppState()
    
    audio_buffer = io.BytesIO()
    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")
    
    # Transcribe using OpenAI Whisper (replace with your transcription logic)
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        response_format="text"
    )

    # Generate bot response
    response_text, new_history = chatbot_response(transcript, state.conversation)

    return None, AppState(conversation=new_history), new_history


# ----------------Initialize the ChatGroq model with LLaMA model parameters-----------    
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",  
    model="llama-3.1-8b-instant",  # Specify the LLaMA model
    timeout=None,
    max_retries=2,
    # Add any additional parameters if needed
)

# -------------------- 🔍 Qdrant Search --------------------
def embed_text(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def search_qdrant(query, top_k=5):
    query_vector = embed_text(query)
    hits = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k)
    return [hit.payload for hit in hits]

# -------------------- 🧠 LLM Reasoning via Groq --------------------
def generate_response_with_llm(query, product_infos, history=[]):
    context = "\n".join([f"- {p['원본상품명']} ({p['가격']}원, {p['원산지']})" for p in product_infos])
    prompt = (
        f"You are an intelligent shopping assistant. A user asked: '{query}'.\n\n"
        f"Here are 5 products retrieved:\n{context}\n\n"
        "Based on the products and user's need, generate a friendly product recommendation summary."
    )
    
    # ✅ Call LLaMA-3.1 8B-Instant via Groq
    res = llm.invoke(prompt)
    return res.content

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
                "상품코드": item.get("productId", f"NAVER-{random.randint(10000,99999)}"),
                "원본상품명": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                "가격": int(item.get("lprice", 0)),
                "배송비용": 2500,
                "원산지": "한국",
                "이미지대URL": item.get("image", "https://via.placeholder.com/150"),
                "Link": item.get("link", "")  # ✅ Added product link
            })
    else:
        print(f"❌ Naver API error: {response.status_code} - {response.text}")

    return results


# -------------------- 🤖 Chatbot Pipeline --------------------
chat_history = []
chat_state = {
    "waiting_for_followup_reply": False,
    "last_query": "",
    "last_ownerclan_results": [],
    "last_naver_results": [],
}

def chatbot_response(user_input, history):
    global chat_state
    history = history[-5:] if history else []

    # CASE 1: User replied to follow-up question → Show rest 2 OC + 2 Naver
    if chat_state["waiting_for_followup_reply"]:
        ownerclan_rest2 = chat_state["last_ownerclan_results"][3:5]
        naver_results = chat_state["last_naver_results"]  # Already fetched earlier
        final_products = ownerclan_rest2 + naver_results

        reasoning_response = generate_response_with_llm(user_input, final_products, history)

        product_display = ""
        for p in final_products:
            product_display += (
                f"**🆔 상품코드**: {p['상품코드']}\n"
                f"**📦 상품명**: {p['원본상품명']}\n"
                f"**💰 가격**: {p['가격']}원\n"
                f"**🚚 배송비용**: {p['배송비용']}원\n"
                f"**🌍 원산지**: {p['원산지']}\n"
            )
            # Show 'View Product' only if a link is available
            product_link = p.get("Link")
            if product_link:
                product_display += f"**🔗 [제품 보기]({product_link})**\n"

            image_url = p.get("이미지대URL", "").strip()
            if image_url and image_url.startswith("http"):
                product_display += f"![image]({image_url})\n\n"
            else:
                product_display += "🖼️ *(No image available)*\n\n"
            product_display += "---\n"

        chat_state["waiting_for_followup_reply"] = False
        return reasoning_response + "\n\n" + product_display, history + [(user_input, product_display)]

    # Combine user history
    combined_query = " ".join([x[0] for x in history]) + " " + user_input
    ownerclan_results = search_qdrant(combined_query, top_k=5)
    ownerclan_top3 = ownerclan_results[:3]
    naver_results = fetch_naver_products(user_input, limit=2)
    all_products = ownerclan_top3 + naver_results

    # CASE 2: User said "no" to initial set → Ask follow-up question only
    if "no" in user_input.lower():
        follow_up_question = random.choice([
            "🔍 Do you prefer a specific brand or color?",
            "🎨 Would you like to explore a different style or category?",
            "🏷️ Are you looking for products in a different price range?"
        ])
        # Save state for next round
        chat_state["waiting_for_followup_reply"] = True
        chat_state["last_query"] = user_input
        chat_state["last_ownerclan_results"] = ownerclan_results
        chat_state["last_naver_results"] = fetch_naver_products(user_input, limit=2)  # new query if needed
        return follow_up_question, history + [(user_input, follow_up_question)]

    # CASE 3: Normal initial recommendation
    reasoning_response = generate_response_with_llm(user_input, all_products, history)

    product_display = ""
    for p in all_products:
        product_display += (
            f"**🆔 상품코드**: {p['상품코드']}\n"
            f"**📦 상품명**: {p['원본상품명']}\n"
            f"**💰 가격**: {p['가격']}원\n"
            f"**🚚 배송비용**: {p['배송비용']}원\n"
            f"**🌍 원산지**: {p['원산지']}\n"
        )
        # Show 'View Product' only if a link is available
        product_link = p.get("Link")
        if product_link:
            product_display += f"**🔗 [제품 보기]({product_link})**\n"

        image_url = p.get("이미지대URL", "").strip()
        if image_url and image_url.startswith("http"):
            product_display += f"![image]({image_url})\n\n"
        else:
            product_display += "🖼️ *(No image available)*\n\n"
        product_display += "---\n"

    product_display += "\n😊 Do these products meet your needs? Reply 'No' if you'd like different suggestions."

    return reasoning_response + "\n\n" + product_display, history + [(user_input, product_display)]


# -------------------- Login/Signup Data --------------------
def on_login(value):
        if value:
            return f"✅ Logged in as: {value}"
        else:
            return "❌ Not logged in."
        
def login(username, password):
    if username == "moni097" and password == "123":
        return "Login successful"
    else:
        return "Login failed"
        
# -------------------- 🎨 Gradio UI --------------------
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .sidebar {
        width: 240px;
        background: linear-gradient(to bottom right, #f8f9fa, #e0e0e0);
        padding: 15px;
        height: 100vh;
        overflow-y: auto;
        border-right: 2px solid #ccc;
        font-size: 14px;
        position: fixed;
        left: 0;
        top: 0;
        z-index: 1000;
        transition: all 0.3s ease-in-out;
    }

    .sidebar h3 {
        margin-bottom: 10px;
        font-size: 18px;
        border-bottom: 1px solid #bbb;
        padding-bottom: 5px;
    }

    .sidebar ul {
        list-style-type: none;
        padding-left: 0;
    }

    .sidebar li {
        padding: 8px 5px;
        margin: 4px 0;
        background-color: #fff;
        border-radius: 6px;
    }

    .sidebar.hidden {
        display: none;
    }

    .toggle-btn {
        position: fixed;
        left: 10px;
        top: 10px;
        z-index: 1100;
        font-size: 22px;
        padding: 8px 12px;
        background-color: #4F46E5;
        color: white;
        border-radius: 6px;
        cursor: pointer;
    }

    .toggle-btn:hover {
        background-color: #3B3BD2;
    }
               
    #naver-shopping-btn {
    background-color: #03C75A;  /* Naver green */
    color: white;
    font-weight: bold;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
    }
    #naver-shopping-btn:hover {
        background-color: #028a3c;
    } 
                         
""") as demo:

    # Add toggle button + static sidebar HTML
    gr.HTML("""
        <div class='toggle-btn' onclick='
            document.getElementById("sidebar").classList.toggle("hidden");
        '>☰ Chat History</div>

        <div id='sidebar' class='sidebar hidden'>
            <h3>🕘Today</h3>
            <ul>
                <li>🗂️ bag</li>
                <li>🗂️ winter shoes price less than 20000 won</li>
                <li>🗂️ 추천 백팩</li>
            </ul>
        </div>
    """)

    # Row with space between left (empty) and right-aligned login button
    with gr.Row():
        with gr.Column(scale=1):  # Takes most of the row, pushes button right
            pass
        with gr.Column(scale=0):
            login_button = gr.LoginButton(
                value="🔐 Sign in",
                logout_value="🚪({})",
                size="sm",
                variant="primary",
                icon="Final Project Code Chatbot/icons8-user.gif",
            )
    login_status = gr.Textbox(visible=False)

    login_button.click(
        fn=on_login,
        inputs=login_button,
        outputs=login_status
    )
    # Main body of the Chatbot 
    gr.Markdown("## 🛍️ Smart Picks: Your AI-Powered Shopping Buddy 🤖")
    gr.Markdown(
    """
    🎉 **Welcome to the AI-Powered Product Recommendation Assistant!** 🛍️🤖🧠

    Ask me anything, like:

    - 👜 *"여름용 여성 백팩 추천해줘"*  
    - 💼 *"Affordable winter bags"*

    👉 The chatbot will recommend the **TOP 5 most relevant products**,  
    complete with **product details** and 🖼️ **image previews !**.

    💬 You can also refine results based on your **preferences** such as brand, price, or color!
    """
    )
    # #---------------1. If Register button enabled-------------------------------------------------------
    # username = gr.Textbox(label="Username (Email Address)")
    # password = gr.Textbox(label="Password", type="password")
    # login_btn = gr.Button("Register")
    # output = gr.Textbox(label="Status")

    # login_btn.click(fn=login, inputs=[username, password], outputs=output)
    #---------------2. If Register button enabled then comment the rest of the below code-------------------------------------------------------
    chatbot = gr.Chatbot(height=520)
    with gr.Row():
        user_input = gr.Textbox(label="Ask about products", placeholder="e.g., 여성 여름 원피스 추천해줘 or Affordable winter bags")
        submit_btn = gr.Button("🔍 Ask")
    with gr.Row():
        input_audio = gr.Audio(
            label="🎙️ Speak your query",
            sources="microphone",
            type="numpy"
        )
        output_audio = gr.Audio(label="Assistant's voice (optional)", visible=False)
    # Naver Shopping Only# 
    def run_naverapi():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjust this path relative to where your script is
        naverapi_path = os.path.join(current_dir, "naverapi.py")  # or adjust the folder if needed

        python_exe = sys.executable  # Use the running python interpreter
        subprocess.Popen([python_exe, naverapi_path])
        return "Naver Shopping app started!"
    
    naver_button = gr.Button("Naver Shopping", elem_id="naver-shopping-btn")
    # Optional: show a confirmation text output when clicked
    output_text = gr.Textbox(value="", interactive=False)
    naver_button.click(fn=run_naverapi, inputs=None, outputs=output_text)
    # Naver Shopping Only# 
    state = gr.State(value=AppState())

    stream = input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=0.5,
        time_limit=10,
    )

    respond = input_audio.stop_recording(
        response,
        [state],
        [output_audio, state, chatbot]
    )
    clear_btn = gr.Button("🧹 Clear Chat")

    def handle_input(user_input, history):
        response, updated_history = chatbot_response(user_input, history)
        return updated_history

    submit_btn.click(handle_input, [user_input, chatbot], chatbot)
    user_input.submit(handle_input, [user_input, chatbot], chatbot)
    clear_btn.click(lambda: [], None, chatbot)

demo.launch()

# bag 
# winter shoes price less than 20000 won
# https://www.gradio.app/guides/conversational-chatbot