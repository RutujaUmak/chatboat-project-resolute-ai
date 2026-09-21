import base64
import io
import json
import os
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from pypdf import PdfReader
from docx import Document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NOVA AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 50%, #172554 100%);
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #ffffff !important;
    }

    .subtitle {
        text-align: center;
        color: #d1d5db !important;
        margin-bottom: 2rem;
        font-weight: 600 !important;
    }

    .status-box {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
    }

    .feature-card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 10px;
    }

    .small-text {
        color: #d1d5db !important;
        font-size: 0.85rem;
        font-weight: 600 !important;
    }

    /* ---------- FORCE BOLD WHITE TEXT IN MAIN AREA ---------- */
    .stApp p, .stApp li,
    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6,
    .stApp span, .stApp label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ---------- CHAT MESSAGES ---------- */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        margin-bottom: 10px;
        background: rgba(255, 255, 255, 0.08) !important;
        padding: 14px 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] strong {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }

    /* ---------- SIDEBAR (light background, dark text) ---------- */
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
        font-weight: 600 !important;
    }

    /* ---------- CHAT INPUT ---------- */
    textarea, input {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ---------- TABS ---------- */
    button[data-baseweb="tab"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

try:
    OLLAMA_API_KEY = st.secrets["ollama"]["api_key"]
except Exception:
    OLLAMA_API_KEY = ""

OLLAMA_BASE_URL = "https://ollama.com"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_context" not in st.session_state:
    st.session_state.uploaded_context = ""

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

if "image_data" not in st.session_state:
    st.session_state.image_data = None

if "image_name" not in st.session_state:
    st.session_state.image_name = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""


# ============================================================
# HELPER: API HEADERS
# ============================================================

def get_headers():
    return {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# GET OLLAMA CLOUD MODELS
# ============================================================

@st.cache_data(ttl=300)
def get_ollama_models():
    """
    Automatically gets models available to the Ollama Cloud account.
    This prevents the 'llama3.2:latest not found' problem.
    """

    if not OLLAMA_API_KEY:
        return [], "API key is missing."

    try:
        response = requests.get(
            OLLAMA_TAGS_URL,
            headers=get_headers(),
            timeout=30,
        )

        if response.status_code != 200:
            return [], (
                f"Ollama model request failed "
                f"({response.status_code}): {response.text}"
            )

        data = response.json()

        models = []

        for model in data.get("models", []):
            name = model.get("name")

            if name and name not in models:
                models.append(name)

        return models, ""

    except requests.exceptions.RequestException as e:
        return [], f"Connection error: {str(e)}"

    except Exception as e:
        return [], f"Unexpected error: {str(e)}"


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file):
    try:
        reader = PdfReader(file)
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages).strip()

    except Exception as e:
        return f"[PDF extraction error: {e}]"


def extract_docx_text(file):
    try:
        document = Document(file)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs).strip()

    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".pdf"):
            return extract_pdf_text(uploaded_file)

        if filename.endswith(".docx"):
            return extract_docx_text(uploaded_file)

        if filename.endswith((".txt", ".md")):
            return uploaded_file.getvalue().decode(
                "utf-8",
                errors="ignore",
            )

        if filename.endswith(".json"):
            raw = uploaded_file.getvalue().decode(
                "utf-8",
                errors="ignore",
            )

            data = json.loads(raw)

            return json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )

        if filename.endswith(".csv"):
            dataframe = pd.read_csv(uploaded_file)

            return dataframe.to_csv(index=False)

        return ""

    except Exception as e:
        return f"[File reading error: {e}]"


# ============================================================
# OCR
# ============================================================

def perform_ocr(image):
    try:
        import pytesseract

        text = pytesseract.image_to_string(image)

        return text.strip()

    except Exception as e:
        return f"OCR error: {e}"


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# IMAGE SUMMARY
# ============================================================

def analyze_image(model, image, instruction):
    """
    Sends image + instruction to Ollama.
    Works only when the selected model supports vision.
    """

    if not OLLAMA_API_KEY:
        return "Ollama API key is missing."

    try:
        image_base64 = image_to_base64(image)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": instruction,
                    "images": [image_base64],
                }
            ],
            "stream": False,
        }

        response = requests.post(
            OLLAMA_CHAT_URL,
            headers=get_headers(),
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            return (
                f"Ollama image API error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        return (
            data.get("message", {})
            .get("content", "")
            .strip()
        )

    except requests.exceptions.RequestException as e:
        return f"Image API connection error: {e}"

    except Exception as e:
        return f"Image analysis error: {e}"


# ============================================================
# CHAT WITH OLLAMA
# ============================================================

def chat_with_ollama(model, messages, temperature):
    if not OLLAMA_API_KEY:
        return "❌ Ollama API key is missing."

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            headers=get_headers(),
            json=payload,
            timeout=180,
        )

        if response.status_code != 200:
            return (
                f"❌ Ollama API error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        answer = (
            data.get("message", {})
            .get("content", "")
        )

        if not answer:
            return "❌ Ollama returned an empty response."

        return answer.strip()

    except requests.exceptions.Timeout:
        return "❌ Ollama request timed out. Please try again."

    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {e}"

    except Exception as e:
        return f"❌ Unexpected error: {e}"


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(text):
    try:
        from gtts import gTTS

        audio_buffer = io.BytesIO()

        tts = gTTS(text=text, lang="en")

        tts.write_to_fp(audio_buffer)

        audio_buffer.seek(0)

        return audio_buffer

    except Exception:
        return None


# ============================================================
# VOICE TO TEXT
# ============================================================

def voice_to_text(audio_file):
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        audio_bytes = audio_file.read()

        with io.BytesIO(audio_bytes) as audio_stream:
            with sr.AudioFile(audio_stream) as source:
                audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)

        return text

    except Exception as e:
        return f"Voice recognition error: {e}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 NOVA AI")

    st.markdown(
        '<div class="small-text">Your intelligent AI workspace</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # API STATUS
    if OLLAMA_API_KEY:
        st.success("🟢 Ollama API configured")
    else:
        st.error("🔴 Ollama API key missing")

        st.info(
            """
            Add this to Streamlit Secrets:

            [ollama]
            api_key = "YOUR_API_KEY"
            """
        )

    # MODEL LOADING
    if OLLAMA_API_KEY:

        with st.spinner("Loading Ollama models..."):
            available_models, model_error = get_ollama_models()

        if available_models:

            selected_model = st.selectbox(
                "🤖 Ollama Model",
                available_models,
                index=0,
            )

        else:

            st.error("❌ No models available.")

            if model_error:
                st.caption(model_error)

            st.stop()

    else:
        st.stop()

    # TEMPERATURE
    temperature = st.slider(
        "🌡️ Creativity",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
    )

    st.divider()

    # NEW CHAT
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_context = ""
        st.session_state.uploaded_files = []
        st.session_state.ocr_text = ""
        st.session_state.image_data = None
        st.session_state.image_name = ""
        st.session_state.last_response = ""

        st.rerun()

    # CLEAR CHAT
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response = ""

        st.rerun()

    st.divider()

    # FEATURES
    st.markdown("### ✨ Features")

    st.markdown(
        """
        <div class="feature-card">
        💬 AI Chat<br>
        📄 Document Analysis<br>
        🔍 OCR<br>
        🖼️ Image Understanding<br>
        🎤 Voice Input<br>
        🔊 Text-to-Speech<br>
        📥 Download Chat
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 NOVA AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Your intelligent AI assistant</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

chat_tab, files_tab, image_tab, voice_tab = st.tabs(
    [
        "💬 Chat",
        "📄 Documents",
        "🖼️ Image & OCR",
        "🎤 Voice",
    ]
)


# ============================================================
# DOCUMENT TAB
# ============================================================

with files_tab:

    st.subheader("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, CSV, MD or JSON",
        type=["pdf", "docx", "txt", "csv", "md", "json"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        all_text = []

        st.session_state.uploaded_files = [
            file.name for file in uploaded_files
        ]

        for file in uploaded_files:

            text = extract_text_from_file(file)

            if text:
                all_text.append(
                    f"\n\n===== {file.name} =====\n\n{text}"
                )

        st.session_state.uploaded_context = "\n".join(all_text)

        st.success(f"✅ {len(uploaded_files)} file(s) processed.")

        st.info(
            "You can now ask questions about these documents in the Chat tab."
        )

        with st.expander("👀 Preview extracted text"):

            preview = st.session_state.uploaded_context

            if len(preview) > 15000:
                preview = preview[:15000] + "\n\n[Preview truncated]"

            st.text_area(
                "Extracted content",
                preview,
                height=400,
            )

        st.download_button(
            "📥 Download Extracted Text",
            data=st.session_state.uploaded_context,
            file_name="nova_extracted_text.txt",
            mime="text/plain",
        )


# ============================================================
# IMAGE TAB
# ============================================================

with image_tab:

    st.subheader("🖼️ Image Understanding + OCR")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader",
    )

    if uploaded_image:

        image = Image.open(uploaded_image)

        st.session_state.image_name = uploaded_image.name

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption=uploaded_image.name,
                use_container_width=True,
            )

        with col2:

            st.markdown("### 🔍 OCR")

            if st.button("Extract Text", use_container_width=True):

                with st.spinner("Running OCR..."):
                    ocr_result = perform_ocr(image)

                st.session_state.ocr_text = ocr_result

            if st.session_state.ocr_text:

                st.text_area(
                    "Detected Text",
                    st.session_state.ocr_text,
                    height=250,
                )

                st.download_button(
                    "📥 Download OCR Text",
                    data=st.session_state.ocr_text,
                    file_name="nova_ocr.txt",
                    mime="text/plain",
                )

        st.divider()

        st.markdown("### 🧠 AI Image Analysis")

        image_instruction = st.text_area(
            "What should NOVA AI do with this image?",
            value=(
                "Describe this image in detail. "
                "Identify the important objects, text, "
                "layout, and overall meaning."
            ),
            height=100,
        )

        if st.button("✨ Analyze Image", use_container_width=True):

            with st.spinner("NOVA AI is analyzing the image..."):

                image_answer = analyze_image(
                    selected_model,
                    image,
                    image_instruction,
                )

            st.markdown("### 🤖 NOVA AI")

            st.write(image_answer)


# ============================================================
# VOICE TAB
# ============================================================

with voice_tab:

    st.subheader("🎤 Voice Assistant")

    st.write(
        "Record your question and NOVA AI will convert it to text."
    )

    audio_value = st.audio_input("🎤 Record your voice")

    if audio_value:

        st.audio(audio_value)

        if st.button("📝 Convert Voice to Text", use_container_width=True):

            with st.spinner("Converting speech to text..."):

                voice_text = voice_to_text(audio_value)

            if voice_text.startswith("Voice recognition error"):

                st.error(voice_text)

            else:

                st.success("✅ Voice converted successfully.")

                st.text_area(
                    "Recognized text",
                    voice_text,
                    height=120,
                )

                if st.button("💬 Send to NOVA AI", use_container_width=True):

                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": voice_text,
                        }
                    )

                    st.rerun()


# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:

    st.subheader("💬 Chat with NOVA AI")

    # DOCUMENT CONTEXT
    context_message = ""

    if st.session_state.uploaded_context:

        context_message = (
            "\n\nThe user uploaded these documents. "
            "Use them when answering questions:\n\n"
            + st.session_state.uploaded_context
        )

    # DISPLAY CHAT HISTORY
    for message in st.session_state.messages:

        role = message["role"]
        content = message["content"]

        with st.chat_message(role):

            st.markdown(content)

            if role == "assistant":

                audio = text_to_speech(content)

                if audio:
                    st.audio(audio, format="audio/mp3")

    # CHAT INPUT
    user_prompt = st.chat_input("Message NOVA AI...")

    if user_prompt:

        # USER MESSAGE
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # PREPARE MESSAGES
        api_messages = []

        system_prompt = """
You are NOVA AI, a helpful, intelligent and professional AI assistant.

Rules:
- Give clear and accurate answers.
- Use simple explanations when appropriate.
- For technical questions, provide practical steps.
- Do not invent information.
- If the user provides document context, use it.
- If information is not present in the document, clearly say so.
"""

        api_messages.append(
            {
                "role": "system",
                "content": system_prompt + context_message,
            }
        )

        # KEEP CHAT HISTORY
        for message in st.session_state.messages[-20:]:

            api_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

        # AI RESPONSE
        with st.chat_message("assistant"):

            with st.spinner("NOVA AI is thinking..."):

                answer = chat_with_ollama(
                    selected_model,
                    api_messages,
                    temperature,
                )

            st.markdown(answer)

            st.session_state.last_response = answer

            # TTS
            if not answer.startswith("❌"):

                audio = text_to_speech(answer)

                if audio:
                    st.audio(audio, format="audio/mp3")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


# ============================================================
# DOWNLOAD CHAT
# ============================================================

if st.session_state.messages:

    st.divider()

    chat_text = []

    for message in st.session_state.messages:

        role = message["role"].upper()

        chat_text.append(
            f"{role}:\n{message['content']}\n"
        )

    complete_chat = "\n".join(chat_text)

    st.download_button(
        "📥 Download Chat",
        data=complete_chat,
        file_name=(
            "nova_ai_chat_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".txt"
        ),
        mime="text/plain",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <div style="text-align:center;color:#9ca3af;font-weight:600;">
        NOVA AI • Powered by Ollama • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
