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
st.set_page_config(page_title="Jagan AI", page_icon="🛕", layout="wide")

# --- CUSTOM CSS TO FORCE BLACK, BOLD TEXT AND LIGHT MODE ---
st.markdown("""
    <style>
    /* Force Light Mode and black text globally */
    :root, .stApp {
        color-scheme: light !important;
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    /* Make all standard text black and bold */
    p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Fix sidebar text and inputs */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] .stTextInput input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 600 !important;
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Fix chat message text */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Fix chat input placeholder */
    input::placeholder, textarea::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
        font-weight: normal !important;
        opacity: 1 !important;
    }
    
    /* Header Styling */
    .header-container {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #c1272d, #e85d04);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(193, 39, 45, 0.2);
    }
    .header-title {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: bold !important;
        margin: 0;
    }
    .header-subtitle {
        color: #ffffff !important;
        font-size: 1.1rem;
        margin-top: 5px;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🛕 Jagan AI</h1>
        <p class="header-subtitle">ଜୟ ଜଗନ୍ନାଥ | Your Intelligent Odisha Assistant</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    api_key = st.text_input("API Key / Ollama", value="ollama", type="password")
    base_url = st.text_input("Base URL", value="http://localhost:11434/v1")
    model_name = st.text_input("Model", value="qwen2.5-coder:7b")
    
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    uploaded_files = st.file_uploader("Upload .txt files about Odisha", type=["txt"], accept_multiple_files=True)
    
    if st.button("Index Uploaded Files", use_container_width=True) and uploaded_files:
        with st.spinner("Indexing knowledge..."):
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
    
    # Safety: Remove dangerous tools for the web interface!
    if "run_shell" in reg.tools: del reg.tools["run_shell"]
    if "write_file" in reg.tools: del reg.tools["write_file"]
    if "edit_file" in reg.tools: del reg.tools["edit_file"]
    if "git_commit" in reg.tools: del reg.tools["git_commit"]

    return Agent(llm, reg, mem, cfg), store

agent, store = get_agent(api_key, base_url, model_name)
st.session_state["store"] = store

# --- CHAT INTERFACE ---
# Initialize chat history
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = []
    # Welcome Message
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("ନମସ୍କାର! (Namaskar!) I am **Jagan**, your AI assistant. "
                    "I can help you write code, research topics, and learn about Odisha's rich culture. "
                    "How can I help you today?")

# Display chat messages from history
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask Jagan anything (English or ଓଡ଼ିଆ)..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Jagan is thinking..."):
            tool_log = []
            def on_tool(name, args):
                tool_log.append(f"🛠️ Used tool: `{name}`")

            try:
                # Update store reference
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