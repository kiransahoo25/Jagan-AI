import streamlit as st
import os
from pathlib import Path

# Import Jagan's core components
from jagan.config import load_config
from jagan.llm import LLM
from jagan.memory import Memory
from jagan.rag import make_embedder, VectorStore, ingest
from jagan.tools.registry import Registry
from jagan.tools import register_all
from jagan.agent import Agent

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Jagan AI | ଜୟ ଜଗନ୍ନାଥ", page_icon="🛕", layout="wide")

# --- CUSTOM CSS FOR DARK GLOW & ROTATING RING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit default header and footer */
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    
    /* 1. The Background Image with Dark Overlay */
    .stApp {
        background-image: linear-gradient(rgba(10, 10, 15, 0.85), rgba(10, 10, 15, 0.95)), 
                          url('https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Jagannath_Temple_Puri.jpg/1920px-Jagannath_Temple_Puri.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* 2. Force Light Text globally for Dark Mode */
    p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
        color: #f8f9fa !important;
    }
    
    /* 3. The Header with Rotating Ring */
    .header-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        height: 250px;
        margin-bottom: 2rem;
    }

    /* Rotating Ring Animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .glowing-ring {
        position: absolute;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        border: 3px solid transparent;
        border-top: 3px solid #e85d04;
        border-left: 3px solid #c1272d;
        box-shadow: 0 0 25px rgba(232, 93, 4, 0.6), inset 0 0 25px rgba(193, 39, 45, 0.6);
        animation: spin 6s linear infinite;
        z-index: 0;
    }
    
    .glowing-ring-inner {
        position: absolute;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        animation: spin 10s linear infinite reverse;
        z-index: 0;
    }

    .header-text {
        z-index: 1;
        text-align: center;
        background: rgba(0, 0, 0, 0.5);
        padding: 20px 40px;
        border-radius: 50%;
        backdrop-filter: blur(5px);
    }

    .header-title {
        color: #ffffff !important;
        font-size: 3rem;
        font-weight: 800 !important;
        margin: 0;
        text-shadow: 0 0 15px rgba(232, 93, 4, 0.8), 0 0 30px rgba(193, 39, 45, 0.6);
    }
    
    /* 4. Chat Bubbles (Glassmorphism) */
    [data-testid="stChatMessage"] {
        background: rgba(30, 30, 40, 0.6) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        padding: 1.2rem !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stChatMessage"] p {
        color: #e0e0e0 !important;
        font-weight: 400 !important;
        font-size: 1.05rem;
    }

    /* 5. Sidebar (Dark Glass) */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 15, 20, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* 6. Chat Input Glow */
    [data-testid="stChatInput"] {
        border-radius: 25px !important;
        border: 1px solid rgba(232, 93, 4, 0.5) !important;
        background-color: rgba(20, 20, 25, 0.8) !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(232, 93, 4, 0.2) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #e85d04 !important;
        box-shadow: 0 0 25px rgba(232, 93, 4, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER WITH ROTATING RING ---
st.markdown("""
    <div class="header-wrapper">
        <div class="glowing-ring"></div>
        <div class="glowing-ring-inner"></div>
        <div class="header-text">
            <h1 class="header-title">🛕 Jagan AI</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #ffb703 !important; font-weight: 600; letter-spacing: 2px;'>ଜୟ ଜଗନ୍ନାଥ | YOUR INTELLIGENT ODISHA ASSISTANT</p>", unsafe_allow_html=True)

# --- HIDDEN CONFIGURATION ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY", "")
    
base_url = "https://api.groq.com/openai/v1"
model_name = "openai/gpt-oss-120b"

# --- SIDEBAR (Knowledge Base Only) ---
with st.sidebar:
    st.markdown("### 🛕 Jagan AI")
    st.markdown("---")
    
    with st.expander("📚 Odisha Knowledge Base", expanded=False):
        st.info("Upload .txt files to teach Jagan about Odisha.")
        uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True, label_visibility="collapsed")
        
        if st.button("Index Files", use_container_width=True) and uploaded_files:
            with st.spinner("Indexing..."):
                data_dir = Path("odisha_knowledge")
                data_dir.mkdir(exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    file_path = data_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                if st.session_state.get("store"):
                    ingest(data_dir, st.session_state["store"])
                    st.success(f"Indexed {len(uploaded_files)} files!")
                else:
                    st.error("Vector store not initialized.")

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
    st.error("⚠️ API Key not found. Please configure `GROQ_API_KEY` in your Streamlit Secrets.")
    st.stop()

agent, store = get_agent(api_key, base_url, model_name)
st.session_state["store"] = store

# --- CHAT INTERFACE ---
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = []
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("ନମସ୍କାର! (Namaskar!) I am **Jagan**. "
                    "I can help you write code, research topics, and learn about Odisha's rich culture. "
                    "How can I help you today?")

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Jagan anything (English or ଓଡ଼ିଆ)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Jagan is thinking..."):
            tool_log = []
            def on_tool(name, args):
                tool_log.append(f"🛠️ Used tool: `{name}`")

            try:
                if store:
                    agent.reg.tools["search_docs"].fn.__globals__['store'] = store
                
                response = agent.step(prompt, on_tool)
                
                if tool_log:
                    with st.expander("View Tool Execution"):
                        for log in tool_log:
                            st.write(log)
                            
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")