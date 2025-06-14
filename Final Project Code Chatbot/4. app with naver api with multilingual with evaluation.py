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
# Add this near your other imports
from metrics import *
from metrics_visualization import display_metrics_graphs
import matplotlib
matplotlib.use('Agg')  # Set the backend to avoid GUI issues

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

# -------------------- 📊 Evaluation Tracking --------------------
@dataclass
class EvaluationTracker:
    queries: list = field(default_factory=list)
    y_true: list = field(default_factory=list)  # Actual relevance (1/0)
    y_pred: list = field(default_factory=list)  # Predicted relevance scores
    clicked_items: list = field(default_factory=list)  # Track which items users click
    
    def add_query(self, query, results, clicked_indices=None):
        """Track a new query and its results"""
        self.queries.append(query)
        
        # For demo, simulate relevance (1=relevant, 0=not relevant)
        # In production, you'd track actual user clicks/feedback
        relevance = [1 if i < 2 else 0 for i in range(len(results))]  # First 2 items considered relevant
        self.y_true.append(relevance)
        
        # Simulate predicted scores (higher=more relevant)
        pred_scores = [1.0, 0.9, 0.7, 0.5, 0.3][:len(results)]
        self.y_pred.append(pred_scores)
        
        if clicked_indices:
            self.clicked_items.append(clicked_indices)
    
    def calculate_metrics(self):
        """Calculate all evaluation metrics"""
        if not self.queries:
            return {}
            
        # Flatten the lists for overall metrics
        flat_y_true = [item for sublist in self.y_true for item in sublist]
        flat_y_pred = [item for sublist in self.y_pred for item in sublist]
        binary_y_pred = [1 if score > 0.5 else 0 for score in flat_y_pred]
        
        k_values = [1, 3, 5]  # Different K values to evaluate
        
        metrics = {
            "precision": [precision_at_k(flat_y_true, binary_y_pred, k) for k in k_values],
            "recall": [recall_at_k(flat_y_true, binary_y_pred, k) for k in k_values],
            "ndcg": [ndcg_at_k(flat_y_true, flat_y_pred, k) for k in k_values],
            "map": [mean_average_precision(flat_y_true, flat_y_pred)],
            "mrr": mean_reciprocal_rank(flat_y_true, flat_y_pred),
            "ctr": ctr(len([i for i in self.clicked_items if i]), len(self.queries)*5),
        }
        
        save_metrics(metrics)  # Save to file
        return metrics
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
    
# -------------------- 🎤 Audio Processing Utilities --------------------
def determine_pause(stream, sampling_rate, state):
    """Simple pause detection based on audio energy"""
    if len(stream) == 0:
        return False
    
    # Calculate RMS energy
    energy = np.sqrt(np.mean(stream**2))
    
    # Consider pause if energy < threshold for >1 second
    threshold = 0.01  # Adjust based on your microphone sensitivity
    min_pause_samples = int(sampling_rate * 1.0)  # 1 second
    
    if energy < threshold:
        if len(stream) > min_pause_samples:
            return True
    return False

#-------------------AppState and voice processing function------------------------------
@dataclass
class AppState:
    stream: np.ndarray = field(default_factory=lambda: np.array([]))  # ✅ Fixed
    sampling_rate: int = 44100
    pause_detected: bool = False
    stopped: bool = False
    started_talking: bool = False
    conversation: list = field(default_factory=list)  # ✅ Also fixed for list

def determine_pause(stream, sampling_rate, state):
    # Dummy pause detection — replace with your actual logic
    return len(stream) / sampling_rate > 2  # Treat as pause if >2 seconds

def process_audio(audio: tuple, state: AppState):
    if audio[1] is None:  # Check if audio data is None
        return None, state
        
    if state.stream.size == 0:  # First chunk
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:  # Subsequent chunks
        state.stream = np.concatenate((state.stream, audio[1]))
    
    state.pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.started_talking = True  # Mark as started talking
    
    if state.pause_detected:
        return gr.Audio(visible=False), state
    return None, state

def speaking(audio_bytes):  # Dummy streaming generator
    yield audio_bytes

#------------------------------Response handler for voice-------------------------
def response(state: AppState):
    if not state.started_talking or state.stream.size == 0:
        return None, AppState(), []
    
    try:
        # Convert numpy array to bytes
        if state.stream.dtype == np.float32:
            audio_data = (state.stream * 32767).astype(np.int16).tobytes()
        else:
            audio_data = state.stream.tobytes()
            
        # Create AudioSegment
        segment = AudioSegment(
            audio_data,
            frame_rate=state.sampling_rate,
            sample_width=state.stream.dtype.itemsize,
            channels=1
        )
        
        # Export to WAV
        with io.BytesIO() as audio_buffer:
            segment.export(audio_buffer, format="wav")
            audio_buffer.seek(0)
            
            # Transcribe using Whisper
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", audio_buffer.read(), "audio/wav"),
                response_format="text"
            )
            
        # Generate response
        response_text, new_history = chatbot_response(transcript, state.conversation)
        
        # Reset state
        return None, AppState(conversation=new_history), new_history
        
    except Exception as e:
        print(f"Audio processing error: {str(e)}")
        return None, AppState(), []

# ----------------Initialize the ChatGroq model with LLaMA model parameters-----------    
llm = ChatGroq(
    temperature=0,
    groq_api_key="",  
    model="llama-3.1-8b-instant",  # Specify the LLaMA model
    timeout=10.0,  # Set timeout to 10 seconds
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
    
    # Detect input language and translate to Korean for Naver search
    input_lang = detect_language(user_input)
    query_for_naver = translate(user_input, "ko") if input_lang != "ko" else user_input
    
    # CASE 1: User replied to follow-up question → Show rest 2 OC + 2 Naver
    if chat_state["waiting_for_followup_reply"]:
        ownerclan_rest2 = chat_state["last_ownerclan_results"][3:5]
        naver_results = chat_state["last_naver_results"]
        final_products = ownerclan_rest2 + naver_results

        reasoning_response = generate_response_with_llm(user_input, final_products, history)
        
        # Translate response to user's language if needed
        if input_lang != "en":  # Assuming LLM responds in English
            reasoning_response = translate(reasoning_response, input_lang)

        product_display = ""
        for p in final_products:
            # Create product info in Korean first
            product_info_ko = (
                f"🆔 상품코드: {p['상품코드']}\n"
                f"📦 상품명: {p['원본상품명']}\n"
                f"💰 가격: {p['가격']}원\n"
                f"🚚 배송비용: {p['배송비용']}원\n"
                f"🌍 원산지: {p['원산지']}\n"
            )
            
            # Translate if needed
            product_info = translate(product_info_ko, input_lang) if input_lang != "ko" else product_info_ko
            
            product_display += product_info.replace("\n", "<br/>")
            
            product_link = p.get("Link")
            if product_link:
                product_display += f"<br/>🔗 <a href='{product_link}' target='_blank'>제품 보기</a><br/>"

            image_url = p.get("이미지대URL", "").strip()
            if image_url and image_url.startswith("http"):
                product_display += f"<img src='{image_url}' width='300'/><br/><br/>"
            else:
                product_display += "🖼️ <i>No image available</i><br/><br/>"
            product_display += "<hr/>"

        chat_state["waiting_for_followup_reply"] = False
        return reasoning_response + "<br/><br/>" + product_display, history + [(user_input, product_display)]

    # Combine user history
    combined_query = " ".join([x[0] for x in history]) + " " + user_input
    ownerclan_results = search_qdrant(combined_query, top_k=5)
    ownerclan_top3 = ownerclan_results[:3]
    naver_results = fetch_naver_products(query_for_naver, limit=2)  # Use translated query
    all_products = ownerclan_top3 + naver_results

    # CASE 2: User said "no" to initial set → Ask follow-up question only
    if "no" in user_input.lower():
        follow_up_questions = {
            "en": "🔍 Do you prefer a specific brand or color?",
            "ko": "🔍 특정 브랜드나 색상을 선호하시나요?",
            "ja": "🔍 特定のブランドや色をお好みですか？",
            "th": "🔍 คุณชอบแบรนด์หรือสีเฉพาะไหม?"
        }
        follow_up_question = follow_up_questions.get(input_lang, follow_up_questions["en"])
        
        chat_state["waiting_for_followup_reply"] = True
        chat_state["last_query"] = user_input
        chat_state["last_ownerclan_results"] = ownerclan_results
        chat_state["last_naver_results"] = fetch_naver_products(query_for_naver, limit=2)
        return follow_up_question, history + [(user_input, follow_up_question)]

    # CASE 3: Normal initial recommendation
    reasoning_response = generate_response_with_llm(user_input, all_products, history)
    
    # Translate response to user's language if needed
    if input_lang != "en":  # Assuming LLM responds in English
        reasoning_response = translate(reasoning_response, input_lang)

    product_display = ""
    for p in all_products:
        # Create product info in Korean first
        product_info_ko = (
            f"🆔 상품코드: {p['상품코드']}\n"
            f"📦 상품명: {p['원본상품명']}\n"
            f"💰 가격: {p['가격']}원\n"
            f"🚚 배송비용: {p['배송비용']}원\n"
            f"🌍 원산지: {p['원산지']}\n"
        )
        
        # Translate if needed
        product_info = translate(product_info_ko, input_lang) if input_lang != "ko" else product_info_ko
        
        product_display += product_info.replace("\n", "<br/>")
        
        product_link = p.get("Link")
        if product_link:
            product_display += f"<br/>🔗 <a href='{product_link}' target='_blank'>제품 보기</a><br/>"

        image_url = p.get("이미지대URL", "").strip()
        if image_url and image_url.startswith("http"):
            product_display += f"<img src='{image_url}' width='300'/><br/><br/>"
        else:
            product_display += "🖼️ <i>No image available</i><br/><br/>"
        product_display += "<hr/>"

    # Add language-specific follow-up prompt
    follow_up_prompts = {
        "en": "<br/>😊 Do these products meet your needs? Reply 'No' if you'd like different suggestions.",
        "ko": "<br/>😊 이 상품들이 필요에 맞나요? 다른 제안을 원하시면 '아니요'라고 답변해주세요.",
        "ja": "<br/>😊 これらの商品はご要望に合っていますか？違う提案が欲しい場合は「いいえ」と返信してください。",
        "th": "<br/>😊 สินค้าเหล่านี้ตรงกับความต้องการของคุณหรือไม่? ตอบ 'ไม่' หากคุณต้องการคำแนะนำที่แตกต่างออกไป"
    }
    product_display += follow_up_prompts.get(input_lang, follow_up_prompts["en"])

    return reasoning_response + "<br/><br/>" + product_display, history + [(user_input, product_display)]


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
                <li>🗂️ ชุดฤดูหนาวสำหรับผู้หญิง</li>
                <li>🗂️ ชุดเดรสกันหนาวสำหรับผู้หญิง </li>
                <li>🗂️ รองเท้าผ้าใบชาย ราคาถูก</li>
                <li>🗂️ black color bag for girls</li> 
                <li>🗂️ winter bag for women</li>
                <li>🗂️ winter shoes price less than 40000 won</li>
                <li>🗂️ 추천 백팩</li>
                <li>🗂️ mens stylish backpack</li>
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
    with gr.Accordion("🌍 Language Options", open=False):
        language_selector = gr.Dropdown(
            choices=["English", "한국어", "日本語", "ไทย"],
            value="English",
            label="Select Interface Language"
        )

    gr.Markdown("""
    🎉 **Welcome to the AI-Powered Product Recommendation Assistant!** 🛍️🤖🧠

    Ask me anything, like:
    - 👜 *"여름용 여성 백팩 추천해줘"*  - 💼 *"Affordable winter bags"* - 👗 *"夏向けのワンピースを教えて"* - 👠 *"แนะนำรองเท้าส้นสูงสำหรับงานแต่งงาน"*
    
    👉 The chatbot will recommend the **TOP 5 most relevant products**, complete with **product details** and 🖼️ **image previews!**  
    
    💬 You can also refine results based on your **preferences** such as brand, price, or color!
    """)
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
            type="numpy",
            streaming=True,
            show_download_button=False
        )
        output_audio = gr.Audio(label="Assistant's response", visible=False)

    state = gr.State(value=AppState())
    # Event handlers
    input_audio.stream(
        process_audio,
        inputs=[input_audio, state],
        outputs=[input_audio, state],
        show_progress="hidden"
    )

    input_audio.stop_recording(
        response,
        inputs=[state],
        outputs=[output_audio, state, chatbot],
        show_progress="hidden"
    )
    # Naver Shopping Only# 
    def run_naverapi():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjust this path relative to where your script is
        naverapi_path = "Final Project Code Chatbot/naverapi_multilingual.py"  # or adjust the folder if needed

        python_exe = sys.executable  # Use the running python interpreter
        subprocess.Popen([python_exe, naverapi_path])
        return "Naver Shopping app started!"
    
    naver_button = gr.Button("Naver Shopping", elem_id="naver-shopping-btn")
    # Optional: show a confirmation text output when clicked
    output_text = gr.Textbox(value="", interactive=False)
    naver_button.click(fn=run_naverapi, inputs=None, outputs=output_text)
    # Naver Shopping Only# 
    # # Add evaluation button and output---------------------#
    with gr.Accordion("📊 Evaluation Metrics", open=False):
        eval_button = gr.Button("Run Evaluation")
        eval_output = gr.Textbox(label="Evaluation Summary", interactive=False)
        eval_plots = gr.Plot(label="Performance Metrics")
    # Track evaluation data globally
    eval_tracker = gr.State(value=EvaluationTracker())
    # Modified chatbot response function to track evaluation data
    def chatbot_response_with_tracking(user_input, history, eval_tracker):
        response, new_history = chatbot_response(user_input, history)
        
        # Simulate getting 5 results (replace with actual results from your system)
        results = [{"product": f"Product {i}"} for i in range(5)]
        eval_tracker.add_query(user_input, results)
        
        return response, new_history, eval_tracker
    def handle_input(user_input, history, eval_tracker):
        response, updated_history, tracker = chatbot_response_with_tracking(user_input, history, eval_tracker)
        return updated_history, tracker
    
    def run_evaluation(eval_tracker):
        metrics = eval_tracker.calculate_metrics()
        
        if not metrics:
            return "No evaluation data available yet. Chat with the bot first.", None
        
        # Generate summary text
        summary = (
            f"📊 Evaluation Results (based on {len(eval_tracker.queries)} queries):\n"
            f"• Mean Average Precision (MAP): {metrics['map'][0]:.3f}\n"
            f"• Mean Reciprocal Rank (MRR): {metrics['mrr']:.3f}\n"
            f"• Click-Through Rate (CTR): {metrics['ctr']:.3f}\n"
            f"• Precision@5: {metrics['precision'][2]:.3f}\n"
            f"• Recall@5: {metrics['recall'][2]:.3f}\n"
            f"• NDCG@5: {metrics['ndcg'][2]:.3f}\n"
        )
        
        # Generate plots
        fig = display_metrics_graphs(metrics)
        if fig is None:
            return summary + "\n\n⚠️ Not enough data to generate plots", None
        
        return summary, fig
    
    clear_btn = gr.Button("🧹 Clear Chat")

    # Update your existing button handlers
    submit_btn.click(
        handle_input, 
        [user_input, chatbot, eval_tracker], 
        [chatbot, eval_tracker]
    )
    
    user_input.submit(
        handle_input, 
        [user_input, chatbot, eval_tracker], 
        [chatbot, eval_tracker]
    )
    
    eval_button.click(
        run_evaluation,
        [eval_tracker],
        [eval_output, eval_plots]
    )
    clear_btn.click(lambda: [], None, chatbot)

demo.launch()

# bag 
# winter shoes price less than 20000 won
# https://www.gradio.app/guides/conversational-chatbot

# - Precision@K: % of top-K results that are relevant
# - Recall@K: % of all relevant items found in top-K
# - NDCG@K: Ranking quality considering position weights
# - MAP: Mean of average precision across queries
# - MRR: Reciprocal rank of first relevant result

# galaxy smart watch price 100000 won
