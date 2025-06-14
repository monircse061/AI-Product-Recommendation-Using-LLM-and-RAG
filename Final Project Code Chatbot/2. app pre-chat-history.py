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
from pydub import AudioSegment

# -------------------- 🔧 Configuration --------------------
OPENAI_API_KEY = ""
QDRANT_URL = ""
QDRANT_API_KEY = ""
COLLECTION_NAME = "products"
VECTOR_DIM = 1536

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
    groq_api_key="",  
    model="llama-3.1-8b-instant",  # Specify the LLaMA model
    timeout=None,
    max_retries=2,
    # Add any additional parameters if needed
)
# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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

# -------------------- 🛒 Naver Mock Recommendations --------------------
def fetch_naver_products(query, limit=2):
    return [{
        "상품코드": f"NAVER-{random.randint(10000,99999)}",
        "원본상품명": f"Naver {query} Product {i+1}",
        "가격": random.randint(10000, 40000),
        "배송비용": 2500,
        "원산지": "한국",
        "이미지대URL": "https://via.placeholder.com/150"
    } for i in range(limit)]

# -------------------- 🤖 Chatbot Pipeline --------------------
chat_history = []

def chatbot_response(user_input, history):
    # Keep recent 5 messages
    history = history[-5:] if history else []

    # Combine history for contextual search
    combined_query = " ".join([x[0] for x in history]) + " " + user_input
    ownerclan_results = search_qdrant(combined_query, top_k=5)

    # Case 1: User says "no" → Show remaining 2 products
    if "no" in user_input.lower():
        ownerclan_rest2 = ownerclan_results[3:5]

        # Random probing question
        follow_up_question = random.choice([
            "🔍 Do you prefer a specific brand or color?",
            "🎨 Would you like to explore a different style or category?",
            "🏷️ Are you looking for products in a different price range?"
        ])
        # just reply in the chatbot one of the follow_up_question
        # then user say something about the follow_up_question then fetch the rest 2 product show.
        # Format rest 2 products
        product_display =  ""
        for p in ownerclan_rest2:
            product_display += (
                f"**🆔 상품코드**: {p['상품코드']}\n"
                f"**📦 상품명**: {p['원본상품명']}\n"
                f"**💰 가격**: {p['가격']}원\n"
                f"**🚚 배송비용**: {p['배송비용']}원\n"
                f"**🌍 원산지**: {p['원산지']}\n"
            )
            image_url = p.get("이미지대URL", "").strip()
            if image_url and image_url.startswith("http"):
                product_display += f"![image]({image_url})\n\n"
            else:
                product_display += "🖼️ *(No image available)*\n\n"
            product_display += "---\n"
        return product_display, history
        reasoning_response = generate_response_with_llm(user_input, product_display, history)
    # Case 2: Initial response → Show top 3 OwnerClan + 2 from Naver
    else:
        ownerclan_top3 = ownerclan_results[:3]
        # Naver Mock API
        naver_results = fetch_naver_products(user_input, limit=2)
        # Final product pool
        all_products = ownerclan_top3 + naver_results

        reasoning_response = generate_response_with_llm(user_input, all_products, history)

        # Format first 5 products
        product_display = ""
        for p in all_products:
            product_display += (
                f"**🆔 상품코드**: {p['상품코드']}\n"
                f"**📦 상품명**: {p['원본상품명']}\n"
                f"**💰 가격**: {p['가격']}원\n"
                f"**🚚 배송비용**: {p['배송비용']}원\n"
                f"**🌍 원산지**: {p['원산지']}\n"
            )
            image_url = p.get("이미지대URL", "").strip()
            if image_url and image_url.startswith("http"):
                product_display += f"![image]({image_url})\n\n"
            else:
                product_display += "🖼️ *(No image available)*\n\n"
            product_display += "---\n"

        product_display += "😊 Do these products meet your needs? Reply 'No' if you'd like different suggestions."
    # Add to history
    history.append((user_input, reasoning_response))
    # Combine reasoning and product display
    full_response = reasoning_response + "\n\n" + product_display
    return full_response, history + [(user_input, full_response)]


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


# 1. Initial interaction:

# Fetch top 5 from OwnerClan → display top 3

# Fetch 2 from Naver

# Save the remaining 2 OwnerClan + 2 Naver for later

# Display LLM-generated response + formatted product cards (3 OwnerClan + 0 Naver shown yet)

# 2. If user replies “no”:

# 🤖 Ask 1 random follow-up question

# 3. Wait for user reply

# When user answers the follow-up:

# Combine:

# Rest 2 OwnerClan

# 2 Naver products

# Call generate_response_with_llm(...) again with new input

# Show the response and all 4 products (2 OC + 2 Naver)
