import base64
import io
import json
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
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #111827 50%,
            #172554 100%
        );
    }

    .nova-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-top: 10px;
    }

    .nova-subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OLLAMA CONFIG
# ============================================================

try:
    OLLAMA_API_KEY = st.secrets["ollama"]["api_key"]
except Exception:
    OLLAMA_API_KEY = ""

OLLAMA_BASE_URL = "https://ollama.com"

OLLAMA_TAGS_URL = (
    f"{OLLAMA_BASE_URL}/api/tags"
)

OLLAMA_CHAT_URL = (
    f"{OLLAMA_BASE_URL}/api/chat"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_context" not in st.session_state:
    st.session_state.document_context = ""

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "free_models" not in st.session_state:
    st.session_state.free_models = []

if "paid_models" not in st.session_state:
    st.session_state.paid_models = []


# ============================================================
# HEADERS
# ============================================================

def ollama_headers():
    return {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# GET MODELS
# ============================================================

@st.cache_data(ttl=300)
def get_models():

    if not OLLAMA_API_KEY:
        return [], "API key missing."

    try:

        response = requests.get(
            OLLAMA_TAGS_URL,
            headers=ollama_headers(),
            timeout=30,
        )

        if response.status_code != 200:
            return [], (
                f"Ollama returned "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        models = []

        for item in data.get("models", []):

            name = item.get("name")

            if name and name not in models:
                models.append(name)

        return models, ""

    except Exception as e:

        return [], str(e)


# ============================================================
# TEST MODEL
# ============================================================

def test_model(model):

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply only with OK."
            }
        ],
        "stream": False,
    }

    try:

        response = requests.post(
            OLLAMA_CHAT_URL,
            headers=ollama_headers(),
            json=payload,
            timeout=60,
        )

        if response.status_code == 200:

            data = response.json()

            answer = (
                data.get("message", {})
                .get("content", "")
            )

            if answer:
                return True, ""

            return False, "Empty response."

        if response.status_code == 402:

            return (
                False,
                "Paid model / credits required."
            )

        if response.status_code == 404:

            return (
                False,
                "Model not found."
            )

        return (
            False,
            f"HTTP {response.status_code}"
        )

    except Exception as e:

        return False, str(e)


# ============================================================
# FIND FREE MODEL
# ============================================================

def find_free_models(models):

    free_models = []
    paid_models = []

    for model in models:

        success, error = test_model(model)

        if success:
            free_models.append(model)

        elif "Paid model" in error:
            paid_models.append(model)

    return free_models, paid_models


# ============================================================
# CHAT
# ============================================================

def ask_ollama(
    model,
    messages,
    temperature,
):

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        },
    }

    try:

        response = requests.post(
            OLLAMA_CHAT_URL,
            headers=ollama_headers(),
            json=payload,
            timeout=180,
        )

        if response.status_code == 402:

            return (
                "❌ This model requires paid Ollama "
                "usage credits. Please select a "
                "free model."
            )

        if response.status_code == 404:

            return (
                "❌ This model is not available. "
                "Please refresh the model list."
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

        return (
            "❌ Request timed out. "
            "Please try again."
        )

    except Exception as e:

        return f"❌ Connection error: {e}"


# ============================================================
# PDF
# ============================================================

def read_pdf(file):

    try:

        reader = PdfReader(file)

        text = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

        return "\n".join(text)

    except Exception as e:

        return f"PDF error: {e}"


# ============================================================
# DOCX
# ============================================================

def read_docx(file):

    try:

        document = Document(file)

        text = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                text.append(paragraph.text)

        return "\n".join(text)

    except Exception as e:

        return f"DOCX error: {e}"


# ============================================================
# FILE EXTRACTION
# ============================================================

def read_file(file):

    name = file.name.lower()

    try:

        if name.endswith(".pdf"):
            return read_pdf(file)

        if name.endswith(".docx"):
            return read_docx(file)

        if name.endswith(".txt"):

            return file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        if name.endswith(".md"):

            return file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        if name.endswith(".json"):

            data = json.loads(
                file.getvalue().decode(
                    "utf-8",
                    errors="ignore"
                )
            )

            return json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )

        if name.endswith(".csv"):

            df = pd.read_csv(file)

            return df.to_csv(index=False)

        return ""

    except Exception as e:

        return f"File error: {e}"


# ============================================================
# OCR
# ============================================================

def run_ocr(image):

    try:

        import pytesseract

        result = pytesseract.image_to_string(
            image
        )

        return result.strip()

    except Exception as e:

        return f"OCR error: {e}"


# ============================================================
# IMAGE BASE64
# ============================================================

def image_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    model,
    image,
    question
):

    image_data = image_base64(image)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question,
                "images": [image_data],
            }
        ],
        "stream": False,
    }

    try:

        response = requests.post(
            OLLAMA_CHAT_URL,
            headers=ollama_headers(),
            json=payload,
            timeout=180,
        )

        if response.status_code == 402:

            return (
                "❌ This model requires paid "
                "usage credits."
            )

        if response.status_code != 200:

            return (
                f"❌ Image API error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        return (
            data.get("message", {})
            .get("content", "")
            .strip()
        )

    except Exception as e:

        return f"❌ Image analysis error: {e}"


# ============================================================
# TEXT TO SPEECH
# ============================================================

def make_speech(text):

    try:

        from gtts import gTTS

        audio = io.BytesIO()

        speech = gTTS(
            text=text,
            lang="en"
        )

        speech.write_to_fp(audio)

        audio.seek(0)

        return audio

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

        audio_stream = io.BytesIO(
            audio_bytes
        )

        with sr.AudioFile(
            audio_stream
        ) as source:

            audio = recognizer.record(
                source
            )

        return recognizer.recognize_google(
            audio
        )

    except Exception as e:

        return (
            f"Voice recognition error: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 NOVA AI")

    st.caption(
        "Free Ollama Cloud AI Assistant"
    )

    st.divider()

    if OLLAMA_API_KEY:

        st.success(
            "🟢 Ollama API Connected"
        )

    else:

        st.error(
            "🔴 Ollama API Key Missing"
        )

        st.info(
            """
Add this in Streamlit Secrets:

[ollama]
api_key = "YOUR_API_KEY"
"""
        )

        st.stop()

    # --------------------------------------------------------
    # MODEL DISCOVERY
    # --------------------------------------------------------

    with st.spinner(
        "Finding available models..."
    ):

        all_models, model_error = get_models()

    if not all_models:

        st.error(
            "❌ No Ollama models found."
        )

        if model_error:
            st.caption(model_error)

        st.stop()

    # --------------------------------------------------------
    # FREE MODEL CHECK
    # --------------------------------------------------------

    if not st.session_state.free_models:

        with st.spinner(
            "Checking model access..."
        ):

            free_models, paid_models = (
                find_free_models(
                    all_models
                )
            )

            st.session_state.free_models = (
                free_models
            )

            st.session_state.paid_models = (
                paid_models
            )

    free_models = (
        st.session_state.free_models
    )

    if free_models:

        st.success(
            f"🆓 {len(free_models)} free model(s) found"
        )

        selected_model = st.selectbox(
            "🤖 Free Model",
            free_models,
        )

    else:

        st.error(
            "❌ No free model is available "
            "for your Ollama account."
        )

        st.warning(
            """
Ollama returned 402 for the available
models.

This is an Ollama account/plan limitation.
Changing Python code cannot bypass it.
"""
        )

        st.markdown(
            "### Models detected"
        )

        for model in all_models:

            st.write(
                f"• {model}"
            )

        st.stop()

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    temperature = st.slider(
        "🌡️ Creativity",
        0.0,
        1.5,
        0.7,
        0.1
    )

    st.divider()

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown("### ✨ Features")

    st.write("💬 AI Chat")
    st.write("📄 PDF / DOCX / TXT")
    st.write("📊 CSV / JSON")
    st.write("🔍 OCR")
    st.write("🖼️ Image Analysis")
    st.write("🎤 Voice Input")
    st.write("🔊 Text-to-Speech")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="nova-title">🤖 NOVA AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nova-subtitle">'
    'Your intelligent AI workspace'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

chat_tab, document_tab, image_tab, voice_tab = (
    st.tabs(
        [
            "💬 Chat",
            "📄 Documents",
            "🖼️ Image & OCR",
            "🎤 Voice",
        ]
    )
)


# ============================================================
# DOCUMENT TAB
# ============================================================

with document_tab:

    st.subheader(
        "📄 Document Analysis"
    )

    files = st.file_uploader(
        "Upload your documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "md",
            "json",
        ],
        accept_multiple_files=True,
    )

    if files:

        extracted = []

        for file in files:

            text = read_file(file)

            extracted.append(
                f"\n===== {file.name} =====\n"
                f"{text}"
            )

        st.session_state.document_context = (
            "\n".join(extracted)
        )

        st.success(
            f"✅ {len(files)} file(s) processed."
        )

        with st.expander(
            "👀 View extracted text"
        ):

            st.text_area(
                "Document content",
                st.session_state.document_context[
                    :20000
                ],
                height=400
            )

        st.download_button(
            "📥 Download Extracted Text",
            st.session_state.document_context,
            "nova_documents.txt",
            "text/plain",
        )


# ============================================================
# IMAGE TAB
# ============================================================

with image_tab:

    st.subheader(
        "🖼️ Image Understanding & OCR"
    )

    image_file = st.file_uploader(
        "Upload an image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        key="nova_image"
    )

    if image_file:

        image = Image.open(
            image_file
        )

        st.image(
            image,
            caption=image_file.name,
            use_container_width=True
        )

        st.divider()

        # OCR

        st.markdown("### 🔍 OCR")

        if st.button(
            "Extract Text from Image",
            use_container_width=True
        ):

            with st.spinner(
                "Running OCR..."
            ):

                ocr_result = run_ocr(
                    image
                )

            st.session_state.ocr_text = (
                ocr_result
            )

        if st.session_state.ocr_text:

            st.text_area(
                "OCR Result",
                st.session_state.ocr_text,
                height=250
            )

            st.download_button(
                "📥 Download OCR",
                st.session_state.ocr_text,
                "nova_ocr.txt",
                "text/plain"
            )

        st.divider()

        # IMAGE AI

        st.markdown(
            "### 🧠 AI Image Analysis"
        )

        image_question = st.text_area(
            "Ask about the image",
            value=(
                "Describe this image in detail. "
                "Identify important objects, text, "
                "and the overall meaning."
            ),
            height=100
        )

        if st.button(
            "✨ Analyze Image",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                result = analyze_image(
                    selected_model,
                    image,
                    image_question
                )

            st.markdown("### 🤖 NOVA AI")

            st.write(result)


# ============================================================
# VOICE TAB
# ============================================================

with voice_tab:

    st.subheader(
        "🎤 Voice Assistant"
    )

    audio = st.audio_input(
        "Record your question"
    )

    if audio:

        st.audio(audio)

        if st.button(
            "📝 Convert to Text",
            use_container_width=True
        ):

            with st.spinner(
                "Converting speech..."
            ):

                text = voice_to_text(
                    audio
                )

            if text.startswith(
                "Voice recognition error"
            ):

                st.error(text)

            else:

                st.success(
                    "✅ Speech converted."
                )

                st.session_state.voice_text = (
                    text
                )

                st.text_area(
                    "Recognized text",
                    text,
                    height=100
                )


# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:

    st.subheader(
        "💬 Chat with NOVA AI"
    )

    # Show previous messages

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
            ):

                audio = make_speech(
                    message["content"]
                )

                if audio:

                    st.audio(
                        audio,
                        format="audio/mp3"
                    )

    prompt = st.chat_input(
        "Message NOVA AI..."
    )

    if prompt:

        # User message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):

            st.markdown(prompt)

        # System prompt

        system_prompt = """
You are NOVA AI, a helpful and professional
AI assistant.

Answer clearly and accurately.

If documents are provided, use them to
answer the user's question.

If the answer is not present in the
documents, say that clearly.

For technical questions, give practical
step-by-step explanations.
"""

        # Document context

        if st.session_state.document_context:

            system_prompt += (
                "\n\nDOCUMENT CONTEXT:\n"
                + st.session_state.document_context[
                    :30000
                ]
            )

        api_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Last 20 messages

        for message in (
            st.session_state.messages[-20:]
        ):

            api_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

        # AI response

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "NOVA AI is thinking..."
            ):

                answer = ask_ollama(
                    selected_model,
                    api_messages,
                    temperature
                )

            st.markdown(answer)

            if not answer.startswith(
                "❌"
            ):

                audio = make_speech(
                    answer
                )

                if audio:

                    st.audio(
                        audio,
                        format="audio/mp3"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# DOWNLOAD CHAT
# ============================================================

if st.session_state.messages:

    st.divider()

    chat_content = []

    for message in (
        st.session_state.messages
    ):

        chat_content.append(
            f"{message['role'].upper()}:\n"
            f"{message['content']}\n"
        )

    st.download_button(
        "📥 Download Chat",
        "\n".join(chat_content),
        file_name=(
            "NOVA_AI_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
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
    <div style="text-align:center;color:#6b7280;">
        🤖 NOVA AI • Streamlit • Ollama
    </div>
    """,
    unsafe_allow_html=True,
)
