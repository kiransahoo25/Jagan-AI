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

# --- CUSTOM CSS FOR A CLEAN, PREMIUM LOOK ---
st.markdown("""
    <style>
    /* Import a clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit default header and footer for a cleaner app feel */
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    
    /* Main background color */
    .stApp {
        background-color: #f4f6f8;
    }
    
    /* Force dark, bold text globally */
    p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    /* Chat Message Styling */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        margin-bottom: 1rem !important;
    }
    [data-testid="stChatMessage"] p {
        font-weight: 400 !important; /* Keep chat text normal weight for readability */
        font-size: 1.05rem;
    }
    
    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        border-radius: 20px !important;
        border: 2px solid #e0e0e0 !important;
        background-color: #ffffff !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #e85d04 !important;
    }
    
    /* Header Container */
    .header-container {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #c1272d, #e85d04);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(232, 93, 4, 0.2);
    }
    .header-title {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 800 !important;
        margin: 0;
        letter-spacing: 1px;
    }
    .header-subtitle {
        color: #ffe3d3 !important;
        font-size: 1.1rem;
        margin-top: 5px;
        font-weight: 500 !important;
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

# --- HIDDEN CONFIGURATION ---
# We no longer show these inputs. They are hardcoded for the cloud deployment.
# If you want to test locally, put your key in a .env file or streamlit secrets.
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    # Fallback for local testing (you can type it in the terminal or set an env var)
    api_key = os.getenv("GROQ_API_KEY", "")
    
base_url = "https://api.groq.com/openai/v1"
model_name = "openai/gpt-oss-120b"

# --- SIDEBAR (Knowledge Base Only) ---
with st.sidebar:
    st.markdown("### 🛕 Jagan AI")
    st.markdown("---")
    
    # Put the knowledge base in an expander so it doesn't clutter the UI
    with st.expander("📚 Odisha Knowledge Base (Admin)", expanded=False):
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
    
    # Safety: Remove dangerous tools for the web interface!
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
# Initialize chat history
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = []
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("ନମସ୍କାର! (Namaskar!) I am **Jagan**. "
                    "I can help you write code, research topics, and learn about Odisha's rich culture. "
                    "How can I help you today?")

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
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