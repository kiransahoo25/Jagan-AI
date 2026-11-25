import streamlit as st
import os
import base64
from pathlib import Path
from groq import Groq

# Import Jagan's core components
from jagan.config import load_config
from jagan.llm import LLM
from jagan.memory import Memory
from jagan.rag import make_embedder, VectorStore, ingest
from jagan.tools.registry import Registry
from jagan.tools import register_all
from jagan.agent import Agent

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="JAGAN AI", page_icon="🛕", layout="wide")

# --- HELPER FUNCTION TO LOAD BACKGROUND IMAGE ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return "" 

bg_image_base64 = get_base64_image("bg.jpg")

# --- CUSTOM CSS FOR FUTURISTIC HUD THEME & MOBILE OPTIMIZATION ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Rajdhani:wght@500;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Rajdhani', sans-serif;
        overflow-x: hidden !important;
    }}

    header[data-testid="stHeader"] {{ background: transparent !important; }}
    
    #MainMenu, footer, [data-testid="stToolbar"], .stDeployButton, 
    [data-testid="stAppDeployButton"], [data-testid="manage-app-button"] {{ 
        display: none !important; 
    }}
    
    /* Chat Input Styling */
    [data-testid="stChatInput"] {{
        display: flex !important;
        visibility: visible !important;
        border-radius: 5px !important;
        border: 1px solid rgba(0, 255, 255, 0.5) !important;
        background-color: rgba(5, 10, 20, 0.95) !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: #e85d04 !important;
        box-shadow: 0 0 30px rgba(232, 93, 4, 0.6) !important;
    }}
    
    [data-testid="collapsedControl"] {{
        color: #00ffff !important;
        background: rgba(5, 5, 12, 0.9) !important;
        border: 1px solid #00ffff !important;
        border-radius: 5px !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        z-index: 99999 !important;
        top: 10px !important;
    }}
    
    .stApp {{
        background-color: #050510;
        background-image: 
            linear-gradient(rgba(5, 5, 16, 0.85), rgba(5, 5, 16, 0.95)),
            url('data:image/jpeg;base64,{bg_image_base64}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        overflow-x: hidden;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.05) 1px, transparent 1px);
        background-size: 40px 40px;
        animation: grid-move 10s linear infinite;
        z-index: -1;
        pointer-events: none;
    }}

    @keyframes grid-move {{
        0% {{ transform: translateY(0); }}
        100% {{ transform: translateY(40px); }}
    }}
    
    .hud-wrapper {{
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        height: 350px;
        margin-bottom: 2rem;
    }}

    @keyframes spin-cw {{ 100% {{ transform: rotate(360deg); }} }}
    @keyframes spin-ccw {{ 100% {{ transform: rotate(-360deg); }} }}
    @keyframes pulse {{ 
        0% {{ box-shadow: 0 0 20px #e85d04, inset 0 0 20px #e85d04; }}
        50% {{ box-shadow: 0 0 60px #e85d04, inset 0 0 40px #e85d04; }}
        100% {{ box-shadow: 0 0 20px #e85d04, inset 0 0 20px #e85d04; }}
    }}

    .hud-ring-outer {{
        position: absolute;
        width: 320px; height: 320px;
        border-radius: 50%;
        border: 2px dashed rgba(0, 255, 255, 0.4);
        animation: spin-cw 15s linear infinite;
    }}

    .hud-ring-mid {{
        position: absolute;
        width: 280px; height: 280px;
        border-radius: 50%;
        border: 3px solid transparent;
        border-top: 3px solid #e85d04;
        border-bottom: 3px solid #c1272d;
        animation: spin-ccw 8s linear infinite;
    }}

    .hud-ring-inner {{
        position: absolute;
        width: 240px; height: 240px;
        border-radius: 50%;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: spin-cw 20s linear infinite;
    }}

    .hud-core {{
        position: absolute;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: rgba(232, 93, 4, 0.1);
        animation: pulse 3s infinite alternate;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1;
    }}

    .header-title {{
        font-family: 'Orbitron', sans-serif;
        color: #ffffff !important;
        font-size: 2.8rem;
        font-weight: 800 !important;
        margin: 0;
        text-shadow: 0 0 10px #e85d04, 0 0 20px #c1272d, 0 0 40px #c1272d;
        letter-spacing: 2px;
        text-align: center;
        white-space: nowrap;
    }}

    [data-testid="stChatMessage"] {{
        background: rgba(10, 15, 30, 0.7) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        border-left: 4px solid #e85d04 !important;
        border-radius: 0px 15px 15px 0px !important;
        box-shadow: 0 0 15px rgba(0,0,0,0.8), inset 0 0 10px rgba(0,255,255,0.1) !important;
        margin-bottom: 1.5rem !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease;
    }}

    [data-testid="stChatMessage"]:hover {{
        box-shadow: 0 0 20px rgba(232, 93, 4, 0.4), inset 0 0 20px rgba(0,255,255,0.2) !important;
        border-color: rgba(232, 93, 4, 0.8) !important;
    }}
    
    [data-testid="stChatMessage"] p {{
        color: #d1e8ff !important;
        font-weight: 500 !important;
        font-size: 1.15rem;
        line-height: 1.5;
    }}

    [data-testid="stSidebar"] {{
        background-color: rgba(5, 5, 12, 0.95) !important;
        border-right: 2px solid rgba(0, 255, 255, 0.3);
        box-shadow: 10px 0 30px rgba(0,0,0,0.8);
    }}
    [data-testid="stSidebar"] * {{
        color: #00ffff !important;
        font-family: 'Rajdhani', sans-serif;
    }}
    
    .subtitle {{
        text-align: center; 
        color: #00ffff !important; 
        font-family: 'Orbitron', sans-serif;
        font-size: 1rem;
        letter-spacing: 4px;
        margin-bottom: 3rem;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }}

    @media (max-width: 768px) {{
        .stApp {{ background-attachment: scroll !important; }}
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }}
        .hud-wrapper {{ height: 220px; margin-bottom: 0.5rem; margin-top: 0.5rem; }}
        .hud-ring-outer {{ width: 200px; height: 200px; }}
        .hud-ring-mid {{ width: 170px; height: 170px; }}
        .hud-ring-inner {{ width: 140px; height: 140px; }}
        .hud-core {{ width: 120px; height: 120px; }}
        .header-title {{ font-size: 1.5rem; letter-spacing: 1px; }}
        .subtitle {{ font-size: 0.65rem; letter-spacing: 2px; margin-bottom: 1.5rem; }}
        [data-testid="stChatMessage"] {{ padding: 0.8rem !important; margin-bottom: 0.8rem !important; border-left-width: 3px !important; }}
        [data-testid="stChatMessage"] p {{ font-size: 1rem !important; line-height: 1.4 !important; }}
        [data-testid="stChatInput"] {{ border-radius: 8px !important; margin-bottom: 5px !important; }}
        [data-testid="stChatInput"] textarea {{ font-size: 16px !important; padding: 12px 10px !important; }}
    }}
    </style>
""", unsafe_allow_html=True)

# --- MASSIVE HUD HEADER ---
st.markdown("""
    <div class="hud-wrapper">
        <div class="hud-ring-outer"></div>
        <div class="hud-ring-mid"></div>
        <div class="hud-ring-inner"></div>
        <div class="hud-core">
            <h1 class="header-title">JAGAN AI</h1>
        </div>
    </div>
    <div class="subtitle">ଜୟ ଜଗନ୍ନାଥ | ODISHA INTELLIGENCE SYSTEM</div>
""", unsafe_allow_html=True)

# --- HIDDEN CONFIGURATION ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY", "")
    
base_url = "https://api.groq.com/openai/v1"
TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b" # Model for image processing

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚡ SYSTEM CONFIG")
    st.markdown("---")
    with st.expander("📚 ODISHA DATABASE", expanded=False):
        st.info("Upload .txt files to sync with the core database.")
        uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True, label_visibility="collapsed")
        if st.button("INITIALIZE SYNC", use_container_width=True) and uploaded_files:
            with st.spinner("Syncing data..."):
                data_dir = Path("odisha_knowledge")
                data_dir.mkdir(exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = data_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                if st.session_state.get("store"):
                    ingest(data_dir, st.session_state["store"])
                    st.success(f"SYNCED {len(uploaded_files)} FILES.")

# --- INITIALIZE BACKEND ---
@st.cache_resource
def get_agent(api_key, base_url, model_name):
    os.environ["JAGAN_API_KEY"] = api_key
    os.environ["JAGAN_BASE_URL"] = base_url
    os.environ["JAGAN_MODEL"] = model_name
    os.environ["JAGAN_CONFIRM"] = "0"
    cfg = load_config()
    llm = LLM(cfg)
    mem = Memory(cfg.home / "memory.db")
    try:
        embed = make_embedder(cfg)
        store = VectorStore(cfg.home / "vectors.db", embed)
    except Exception:
        store = None
    reg = Registry()
    register_all(reg, cfg, mem, store)
    if "run_shell" in reg.tools: del reg.tools["run_shell"]
    if "write_file" in reg.tools: del reg.tools["write_file"]
    if "edit_file" in reg.tools: del reg.tools["edit_file"]
    if "git_commit" in reg.tools: del reg.tools["git_commit"]
    return Agent(llm, reg, mem, cfg), store

if not api_key:
    st.error("⚠️ SYSTEM OFFLINE: API Key missing in secrets.")
    st.stop()

agent, store = get_agent(api_key, base_url, TEXT_MODEL)
st.session_state["store"] = store

# --- AUDIO TRANSCRIPTION ---
def transcribe_audio(audio_file, api_key):
    try:
        client = Groq(api_key=api_key)
        file_tuple = ("audio.wav", audio_file.getvalue(), "audio/wav")
        return client.audio.transcriptions.create(
            file=file_tuple,
            model="whisper-large-v3",
            language="or",
            response_format="text"
        )
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        return None

# --- CHAT INTERFACE ---
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = []
    with st.chat_message("assistant", avatar="🛕"):
        st.markdown("**SYSTEM INITIALIZED.** Namaskar, I am Jagan. You can **type**, **speak**, or **upload an image**.")

# Render previous chat history
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🛕"
    with st.chat_message(message["role"], avatar=avatar):
        if isinstance(message["content"], list):
            # Extract text from complex payload
            text_content = next((item["text"] for item in message["content"] if item["type"] == "text"), "")
            st.markdown(text_content)
            if any(item["type"] == "image_url" for item in message["content"]):
                st.caption("📸 Image attached")
        else:
            st.markdown(message["content"])

# --- VISION UPLOADER (Placed right above chat input) ---
with st.popover("📸 Vision (Upload or Capture)"):
    img_file = st.camera_input("Take a photo") or st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])
    if img_file:
        st.session_state["current_image"] = img_file
        st.image(img_file, caption="Image ready to send", use_container_width=True)

# --- CHAT INPUT ---
chat_input_result = st.chat_input("Ask Jagan (English / ଓଡ଼ିଆ)...", accept_audio=True)

if chat_input_result:
    prompt = None
    img_b64 = None
    
    # 1. Process Audio
    if hasattr(chat_input_result, "audio") and chat_input_result.audio is not None:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.audio(chat_input_result.audio)
            with st.spinner("🎙️ Transcribing..."):
                prompt = transcribe_audio(chat_input_result.audio, api_key)
            if prompt: st.markdown(f"_{prompt}_")
                
    # 2. Process Text
    elif hasattr(chat_input_result, "text") and chat_input_result.text:
        prompt = chat_input_result.text
    elif isinstance(chat_input_result, str):
        prompt = chat_input_result

    # 3. Check for Image
    if "current_image" in st.session_state and st.session_state["current_image"] is not None:
        img_bytes = st.session_state["current_image"].getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # If user didn't type anything, give a default prompt
        if not prompt:
            prompt = "Describe this image in detail. If there is text, read it."

    # 4. Execute
    if prompt:
        # Display user bubble (if text/audio hasn't been displayed yet)
        if not (hasattr(chat_input_result, "audio") and chat_input_result.audio):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
                if img_b64: st.image(st.session_state["current_image"], width=200)

        # Build payload
        if img_b64:
            # Format for Vision API
            content_payload = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
            st.session_state.messages.append({"role": "user", "content": content_payload})
            
            # Temporarily switch model to Vision
            agent.llm.model = VISION_MODEL 
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            agent.llm.model = TEXT_MODEL

        # Generate Response
        with st.chat_message("assistant", avatar="🛕"):
            with st.spinner("PROCESSING..."):
                try:
                    if store:
                        agent.reg.tools["search_docs"].fn.__globals__['store'] = store
                    
                    response = agent.step(prompt if not img_b64 else content_payload)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Clean up image and reset model
                    st.session_state["current_image"] = None
                    agent.llm.model = TEXT_MODEL 
                    st.rerun() # Refresh to clear the image uploader
                except Exception as e:
                    st.error(f"SYSTEM ERROR: {str(e)}")