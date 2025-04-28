import gradio as gr
from logic import chatbot_response, signup_user, login_user, get_previous_chats
import json
import os

# -------------------- Global States --------------------
username_state = gr.State(value="")
history_state = gr.State(value=[])
sidebar_visible_state = gr.State(value=True)

# -------------------- Interface Logic --------------------
def handle_input(user_input, history, username):
    if not username:
        return gr.update(visible=True), "", history, history
    response, new_history = chatbot_response(user_input, history, username)
    return gr.update(visible=False), "", new_history, new_history

def toggle_sidebar(visible):
    return not visible

# -------------------- Layout --------------------
with gr.Blocks(theme=gr.themes.Soft(), css="body {font-family: 'Segoe UI';}") as demo:
    with gr.Row():
        sidebar_toggle = gr.Button("📋", size="sm", elem_id="sidebar-btn", scale=0)
        gr.Markdown("## 🛍️ Smart Picks: Your AI-Powered Shopping Buddy 🤖", scale=8)
        with gr.Row(scale=4):
            profile_dropdown = gr.Dropdown(["Login", "Signup"], label=None, scale=3)
            share_btn = gr.Button("🔗 Share", size="sm", scale=1)

    gr.Markdown("""
🎉 **Welcome to the AI-Powered Product Recommendation Assistant!** 🛍️🤖🧠

Ask me anything, like:

- 👜 *"여름용 여성 백팩 추천해줘"*  
- 💼 *"Affordable winter bags"*

👉 The chatbot will recommend the **TOP 5 most relevant products**,  
complete with **product details** and 🖼️ **image previews**.

💬 You can also refine results based on your **preferences** such as brand, price, or color!
""")

    with gr.Row():
        with gr.Column(scale=2, visible=True) as sidebar:
            gr.Markdown("## 💬 Previous Chats")
            prev_chats = gr.JSON(label="")

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(label="Smart Picks Bot", type="messages", height=500)
            with gr.Row():
                user_input = gr.Textbox(label="Ask about products", placeholder="e.g., 여성 여름 원피스 추천해줘", scale=8)
                submit_btn = gr.Button("🔍 Ask", scale=2)
                clear_btn = gr.Button("🧹 Clear", scale=2)

    with gr.Column(visible=False) as login_panel:
        gr.Markdown("🔐 **Login Panel**")
        login_username = gr.Textbox(label="Username")
        login_password = gr.Textbox(label="Password", type="password")
        login_submit = gr.Button("Login")
        login_status = gr.Markdown("")

    # -------------------- Function Bindings --------------------
    submit_btn.click(handle_input, [user_input, history_state, username_state],
                     [login_panel, user_input, chatbot, history_state])
    user_input.submit(handle_input, [user_input, history_state, username_state],
                      [login_panel, user_input, chatbot, history_state])
    clear_btn.click(lambda: [], None, chatbot)

    sidebar_toggle.click(lambda v: gr.update(visible=not v), [sidebar_visible_state], sidebar)

    def handle_profile_action(action):
        if action == "Login":
            return gr.update(visible=True)
        elif action == "Signup":
            return gr.update(visible=True)
        return gr.update(visible=False)

    profile_dropdown.change(handle_profile_action, profile_dropdown, login_panel)

    login_submit.click(
        login_user,
        [login_username, login_password],
        [login_status, username_state, login_panel]
    )

    demo.load(get_previous_chats, [username_state], [prev_chats])

demo.launch()