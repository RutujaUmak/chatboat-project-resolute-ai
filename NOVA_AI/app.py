import streamlit as st
import requests
import json
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="NOVA AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔑 Ollama Cloud API settings
OLLAMA_API_KEY = "35d039b439e24e429473186b126fa3d5.acNSTW1xEeqk2cmmnjlKq1yq"
OLLAMA_URL = "https://ollama.com/api/chat"

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* ---------- APP ---------- */
.stApp {
    background: #ffffff;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: #f7f7f8;
    border-right: 1px solid #e5e5e5;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
.sidebar-logo {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 3px;
}
.sidebar-subtitle {
    color: #6b7280;
    font-size: 12px;
    margin-bottom: 20px;
}

/* ---------- MAIN ---------- */
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    color: #111827;
    margin-top: 25px;
}
.main-subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 30px;
}

/* ---------- WELCOME ---------- */
.welcome-box {
    max-width: 850px;
    margin: 20px auto;
    padding: 25px;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    background: #fafafa;
    text-align: center;
}
.welcome-title {
    font-size: 24px;
    font-weight: 600;
    color: #111827;
}
.welcome-text {
    color: #6b7280;
    margin-top: 8px;
}

/* ---------- FEATURE CARDS ---------- */
.feature {
    padding: 18px;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    background: white;
    min-height: 120px;
}
.feature-icon {
    font-size: 25px;
}
.feature-title {
    font-weight: 600;
    margin-top: 8px;
    color: #111827;
}
.feature-text {
    font-size: 13px;
    color: #6b7280;
}

/* ---------- CHAT ---------- */
[data-testid="stChatMessage"] {
    max-width: 850px;
    margin-left: auto;
    margin-right: auto;
}

/* ---------- INPUT ---------- */
[data-testid="stChatInput"] {
    max-width: 850px;
    margin-left: auto;
    margin-right: auto;
}

/* ---------- BUTTON ---------- */
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
}

/* ---------- FILE ---------- */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 11px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("🤖 **NOVA AI**")
    st.caption("Your intelligent AI workspace")
    st.divider()

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### 🧠 Model")

    model = st.selectbox(
        "Ollama model",
        [
            "gemma4:31b",
            "gpt-oss:20b",
            "gpt-oss:120b",
            "kimi-k2.7-code",
            "deepseek-v4.1-flash",
            "deepseek-v4-flash:0731",
            "glm-5.3-flash",
            "glm-5.3",
            "minimax-m3",
            "nemotron-3-super",
            "qwen3.5:397b",
            "mistral-large-3:675b",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📎 Upload files")

    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, CSV, Images",
        type=[
            "pdf", "docx", "txt", "csv", "md", "json",
            "png", "jpg", "jpeg", "webp"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.files:
                filename = file.name.lower()
                try:
                    if filename.endswith(".pdf"):
                        reader = PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"

                    elif filename.endswith(".docx"):
                        doc = Document(file)
                        text = "\n".join(p.text for p in doc.paragraphs)

                    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                        image = Image.open(file)
                        text = pytesseract.image_to_string(image)

                    else:
                        text = file.read().decode("utf-8", errors="ignore")

                    st.session_state.files[file.name] = text

                except Exception as e:
                    st.session_state.files[file.name] = f"File processing error: {e}"

        st.success(f"{len(uploaded_files)} file(s) uploaded")

    if st.session_state.files:
        st.markdown("### 📚 Your files")
        for filename in st.session_state.files:
            st.caption(f"📄 {filename}")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("🦙 Ollama Cloud • Llama 3.2")
    st.caption("🔒 Secure API Connection")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown('<div class="main-title">🤖 NOVA AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Your private AI assistant powered by Ollama Cloud</div>',
    unsafe_allow_html=True
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-box">
        <div class="welcome-title">👋 Welcome to NOVA AI</div>
        <div class="welcome-text">
        Ask questions, upload documents, analyze images,
        write code, or start a conversation.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Document AI</div>
            <div class="feature-text">
            Upload PDFs and documents and ask questions about them.
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="feature">
            <div class="feature-icon">💻</div>
            <div class="feature-title">Coding Assistant</div>
            <div class="feature-text">
            Generate and explain Python, SQL, C, C++ and more.
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="feature">
            <div class="feature-icon">🎤</div>
            <div class="feature-title">Voice Assistant</div>
            <div class="feature-text">
            Record your voice and interact with your AI assistant.
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# VOICE INPUT
# ============================================================

audio = st.audio_input("🎤 Voice input")

if audio:
    st.info(
        "Voice recording received. "
        "Whisper speech-to-text can be connected next."
    )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("Message NOVA AI...")


# ============================================================
# CHAT PROCESSING
# ============================================================

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ------------------------------------------
    # FILE CONTEXT
    # ------------------------------------------

    file_context = ""

    if st.session_state.files:
        file_context = "\n\nUploaded file information:\n"
        for filename, content in st.session_state.files.items():
            file_context += f"\n--- {filename} ---\n"
            file_context += content[:10000]

    # ------------------------------------------
    # MESSAGES
    # ------------------------------------------

    messages = [
        {
            "role": "system",
            "content":
            """
            You are NOVA AI.

            You are a helpful, intelligent AI assistant.

            Give clear and accurate answers.

            Use Markdown for formatting.

            When writing code, provide complete
            and properly formatted code.

            If uploaded documents are provided,
            use them when answering questions.
            """
            + file_context
        }
    ]

    messages.extend(st.session_state.messages)

    # ------------------------------------------
    # OLLAMA CLOUD API CALL
    # ------------------------------------------

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        try:

            response = requests.post(
                OLLAMA_URL,
                headers={
                    "Authorization": f"Bearer {OLLAMA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=300
            )

            if response.status_code != 200:
                placeholder.error(
                    f"Ollama Cloud error: {response.status_code} - {response.text}"
                )

            else:

                for line in response.iter_lines():

                    if not line:
                        continue

                    data = json.loads(line.decode("utf-8"))

                    content = data.get("message", {}).get("content", "")

                    full_response += content

                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

        except requests.exceptions.ConnectionError:
            placeholder.error(
                "❌ Could not connect to Ollama Cloud. "
                "Please check your internet connection and API key."
            )

        except Exception as e:
            placeholder.error(f"❌ Error: {e}")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">NOVA AI • Ollama Cloud • Secure API</div>',
    unsafe_allow_html=True
)
