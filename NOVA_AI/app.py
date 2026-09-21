import streamlit as st
import requests
import json
import base64
import io
import re
from datetime import datetime

from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

# Voice
import speech_recognition as sr

# Text to speech
from gtts import gTTS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NOVA AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIG
# ============================================================

# IMPORTANT:
# Add these to Streamlit Cloud:
#
# Settings → Secrets
#
# [ollama]
# api_key = "YOUR_NEW_API_KEY"
#
# Do NOT put your API key directly inside this file.

try:
    OLLAMA_API_KEY = st.secrets["ollama"]["api_key"]
except Exception:
    OLLAMA_API_KEY = ""

OLLAMA_URL = "https://ollama.com/api/chat"

DEFAULT_MODEL = "llama3.2:latest"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
========================================================= */

.stApp {
    background: #ffffff;
}

.main .block-container {
    max-width: 1150px;
    padding-top: 1rem;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background: #f7f7f8;
    border-right: 1px solid #e5e7eb;
}

.sidebar-brand {
    font-size: 25px;
    font-weight: 800;
    color: #111827;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #6b7280;
}


/* =========================================================
   HEADER
========================================================= */

.nova-header {
    text-align: center;
    padding: 18px 0 10px 0;
}

.nova-title {
    font-size: 38px;
    font-weight: 800;
    color: #111827;
}

.nova-subtitle {
    color: #6b7280;
    font-size: 14px;
}


/* =========================================================
   WELCOME
========================================================= */

.welcome-box {
    margin: 20px auto;
    max-width: 900px;
    padding: 30px;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    background: linear-gradient(135deg, #fafafa, #ffffff);
    text-align: center;
}

.welcome-title {
    font-size: 25px;
    font-weight: 700;
}

.welcome-text {
    color: #6b7280;
    margin-top: 8px;
}


/* =========================================================
   FEATURE CARDS
========================================================= */

.feature-card {
    border: 1px solid #e5e7eb;
    border-radius: 17px;
    padding: 20px;
    min-height: 145px;
    background: white;
    transition: 0.2s;
}

.feature-card:hover {
    border-color: #cbd5e1;
    transform: translateY(-2px);
}

.feature-icon {
    font-size: 28px;
}

.feature-title {
    font-weight: 700;
    margin-top: 8px;
}

.feature-text {
    font-size: 13px;
    color: #6b7280;
    margin-top: 5px;
}


/* =========================================================
   CHAT
========================================================= */

[data-testid="stChatMessage"] {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}


/* =========================================================
   CHAT INPUT
========================================================= */

[data-testid="stChatInput"] {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}


/* =========================================================
   STATUS
========================================================= */

.status-online {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 8px 12px;
    border-radius: 10px;
    font-size: 13px;
}

.status-offline {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 8px 12px;
    border-radius: 10px;
    font-size: 13px;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 11px;
    margin-top: 35px;
    padding-bottom: 20px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = {}

if "images" not in st.session_state:
    st.session_state.images = {}

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text, max_chars=12000):
    """Limit context size."""
    if not text:
        return ""

    text = str(text)

    if len(text) > max_chars:
        return text[:max_chars] + "\n...[content truncated]"

    return text


def image_to_base64(uploaded_file):
    """Convert image to base64 for Ollama vision models."""

    image = Image.open(uploaded_file)

    # Convert to RGB for compatibility
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def extract_file_text(uploaded_file):
    """Extract text from PDF, DOCX, TXT, CSV, MD, JSON."""

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".pdf"):

            reader = PdfReader(uploaded_file)

            text_parts = []

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text_parts.append(page_text)

            return "\n".join(text_parts)


        elif filename.endswith(".docx"):

            doc = Document(uploaded_file)

            return "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
            )


        elif filename.endswith(
            (".txt", ".csv", ".md", ".json")
        ):

            data = uploaded_file.read()

            return data.decode(
                "utf-8",
                errors="ignore"
            )


        return ""

    except Exception as e:

        return f"File processing error: {e}"


def perform_ocr(uploaded_file):
    """Extract text from image using Tesseract."""

    try:

        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        return text.strip()

    except Exception as e:

        return f"OCR error: {e}"


def speech_to_text(audio_bytes):
    """Convert recorded audio to text."""

    try:

        recognizer = sr.Recognizer()

        audio_file = io.BytesIO(audio_bytes)

        with sr.AudioFile(audio_file) as source:

            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(
            audio_data
        )

        return text

    except sr.UnknownValueError:

        return "I could not understand the audio."

    except sr.RequestError as e:

        return f"Speech recognition service error: {e}"

    except Exception as e:

        return f"Voice processing error: {e}"


def text_to_speech(text):
    """Generate MP3 speech."""

    try:

        if not text:
            return None

        # Avoid extremely large audio requests
        text = text[:4000]

        audio_buffer = io.BytesIO()

        tts = gTTS(
            text=text,
            lang="en"
        )

        tts.write_to_fp(audio_buffer)

        audio_buffer.seek(0)

        return audio_buffer

    except Exception:
        return None


def ask_ollama(
    messages,
    model,
    temperature=0.7,
    image_base64=None
):
    """Call Ollama Cloud."""

    if not OLLAMA_API_KEY:

        return None, (
            "Ollama API key is missing. "
            "Add it to Streamlit Secrets."
        )

    try:

        request_messages = []

        for message in messages:

            new_message = {
                "role": message["role"],
                "content": message["content"]
            }

            request_messages.append(new_message)

        # Add image to latest user message
        if image_base64:

            for message in reversed(request_messages):

                if message["role"] == "user":

                    message["images"] = [
                        image_base64
                    ]

                    break

        response = requests.post(
            OLLAMA_URL,
            headers={
                "Authorization":
                    f"Bearer {OLLAMA_API_KEY}",
                "Content-Type":
                    "application/json"
            },
            json={
                "model": model,
                "messages": request_messages,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            },
            stream=True,
            timeout=300
        )

        if response.status_code != 200:

            return None, (
                f"Ollama API error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        full_response = ""

        for line in response.iter_lines():

            if not line:
                continue

            try:

                data = json.loads(
                    line.decode("utf-8")
                )

                content = data.get(
                    "message",
                    {}
                ).get(
                    "content",
                    ""
                )

                full_response += content

            except Exception:
                continue

        if not full_response:

            return None, "Ollama returned an empty response."

        return full_response, None

    except requests.exceptions.Timeout:

        return None, (
            "Ollama request timed out. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        return None, (
            "Could not connect to Ollama Cloud. "
            "Please check your internet connection."
        )

    except Exception as e:

        return None, f"Unexpected error: {e}"


def build_file_context():
    """Create document context."""

    if not st.session_state.files:
        return ""

    context = "\n\nUPLOADED DOCUMENT CONTEXT:\n"

    for filename, content in st.session_state.files.items():

        context += (
            f"\n--- {filename} ---\n"
            f"{clean_text(content, 8000)}\n"
        )

    return context


def download_chat_text():

    lines = []

    lines.append("NOVA AI CONVERSATION")
    lines.append("=" * 50)
    lines.append("")

    for message in st.session_state.messages:

        role = message["role"].upper()

        lines.append(
            f"{role}:"
        )

        lines.append(
            message["content"]
        )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🤖 NOVA AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Your intelligent AI workspace'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # ========================================================
    # NEW CHAT
    # ========================================================

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.images = {}
        st.session_state.ocr_results = {}
        st.session_state.voice_text = ""

        st.rerun()


    st.divider()


    # ========================================================
    # MODEL
    # ========================================================

    st.markdown("### 🧠 AI Model")

    model = st.selectbox(
        "Choose model",
        [
            "llama3.2:latest",
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
            "mistral-large-3:675b"
        ],
        index=0
    )


    temperature = st.slider(
        "🌡️ Creativity",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1
    )


    st.divider()


    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.markdown("### 📎 Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "md",
            "json"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        for file in uploaded_files:

            if file.name not in st.session_state.files:

                content = extract_file_text(file)

                st.session_state.files[
                    file.name
                ] = content

        st.success(
            f"{len(uploaded_files)} document(s) ready"
        )


    if st.session_state.files:

        st.markdown("#### 📚 Loaded documents")

        for filename in st.session_state.files:

            st.caption(
                f"📄 {filename}"
            )


    st.divider()


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    st.markdown("### 📷 Image AI")

    uploaded_images = st.file_uploader(
        "Upload images",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        accept_multiple_files=True,
        key="image_uploader"
    )

    if uploaded_images:

        for image_file in uploaded_images:

            if image_file.name not in st.session_state.images:

                st.session_state.images[
                    image_file.name
                ] = image_file.getvalue()


    st.divider()


    # ========================================================
    # STATISTICS
    # ========================================================

    st.markdown("### 📊 Statistics")

    st.metric(
        "Questions",
        st.session_state.total_questions
    )

    st.metric(
        "Documents",
        len(st.session_state.files)
    )

    st.metric(
        "Images",
        len(st.session_state.images)
    )


    st.divider()


    # ========================================================
    # DOWNLOAD
    # ========================================================

    if st.session_state.messages:

        st.download_button(
            "📥 Download Chat",
            data=download_chat_text(),
            file_name="nova_ai_chat.txt",
            mime="text/plain",
            use_container_width=True
        )


    # ========================================================
    # CLEAR
    # ========================================================

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()

    if OLLAMA_API_KEY:

        st.markdown(
            '<div class="status-online">'
            '🟢 Ollama API configured'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="status-offline">'
            '🔴 Ollama API key missing'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
<div class="nova-header">
    <div class="nova-title">🤖 NOVA AI</div>
    <div class="nova-subtitle">
        Your intelligent AI workspace
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
<div class="welcome-box">

<div class="welcome-title">
👋 Welcome to NOVA AI
</div>

<div class="welcome-text">
Chat • Documents • OCR • Image AI • Voice • Coding
</div>

</div>
""",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
<div class="feature-card">
<div class="feature-icon">🤖</div>
<div class="feature-title">AI Chat</div>
<div class="feature-text">
Ask questions and have conversations with NOVA.
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
<div class="feature-card">
<div class="feature-icon">📷</div>
<div class="feature-title">Vision + OCR</div>
<div class="feature-text">
Read text and analyze uploaded images.
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
<div class="feature-card">
<div class="feature-icon">📚</div>
<div class="feature-title">Document AI</div>
<div class="feature-text">
Analyze PDFs, DOCX, CSV and text files.
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
<div class="feature-card">
<div class="feature-icon">🎙️</div>
<div class="feature-title">Voice AI</div>
<div class="feature-text">
Speak naturally and send your voice to NOVA.
</div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# IMAGE WORKSPACE
# ============================================================

if st.session_state.images:

    st.divider()

    st.subheader("📷 Image Workspace")

    image_names = list(
        st.session_state.images.keys()
    )

    selected_image = st.selectbox(
        "Select image",
        image_names
    )

    image_bytes = st.session_state.images[
        selected_image
    ]

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    st.image(
        image,
        caption=selected_image,
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)


    # ========================================================
    # OCR
    # ========================================================

    with col1:

        if st.button(
            "🔍 Extract OCR",
            use_container_width=True
        ):

            with st.spinner(
                "Reading text from image..."
            ):

                result = perform_ocr(
                    io.BytesIO(image_bytes)
                )

            st.session_state.ocr_results[
                selected_image
            ] = result


    # ========================================================
    # IMAGE SUMMARY
    # ========================================================

    with col2:

        if st.button(
            "✨ Summarize Image",
            use_container_width=True
        ):

            st.session_state["image_summary_request"] = (
                selected_image
            )

            st.rerun()


    # ========================================================
    # ASK ABOUT IMAGE
    # ========================================================

    with col3:

        if st.button(
            "💬 Ask About Image",
            use_container_width=True
        ):

            st.session_state["image_question_mode"] = (
                selected_image
            )

            st.rerun()


    # ========================================================
    # OCR RESULT
    # ========================================================

    if selected_image in st.session_state.ocr_results:

        st.markdown("### 🔍 OCR Result")

        ocr_text = st.session_state.ocr_results[
            selected_image
        ]

        if ocr_text:

            st.text_area(
                "Extracted text",
                ocr_text,
                height=220
            )

            st.download_button(
                "📥 Download OCR",
                data=ocr_text,
                file_name=(
                    selected_image.rsplit(
                        ".",
                        1
                    )[0]
                    + "_ocr.txt"
                ),
                mime="text/plain"
            )

        else:

            st.warning(
                "No readable text was detected."
            )


    # ========================================================
    # IMAGE SUMMARY
    # ========================================================

    if st.session_state.get(
        "image_summary_request"
    ) == selected_image:

        st.markdown("### ✨ AI Image Summary")

        with st.spinner(
            "NOVA is analyzing the image..."
        ):

            image_b64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")

            vision_messages = [
                {
                    "role": "user",
                    "content": (
                        "Analyze this image carefully. "
                        "Describe the main objects, scene, "
                        "visible text, important details, "
                        "and give a useful concise summary."
                    )
                }
            ]

            summary, error = ask_ollama(
                vision_messages,
                model,
                temperature,
                image_base64=image_b64
            )

        if error:

            st.error(
                "Image analysis failed.\n\n"
                + error
                + "\n\n"
                "Make sure the selected Ollama model "
                "supports vision/image input."
            )

        else:

            st.markdown(summary)

        st.session_state[
            "image_summary_request"
        ] = None


    # ========================================================
    # IMAGE QUESTION
    # ========================================================

    if st.session_state.get(
        "image_question_mode"
    ) == selected_image:

        st.markdown("### 💬 Ask About Image")

        image_question = st.text_input(
            "What would you like to know?"
        )

        if st.button(
            "🚀 Ask NOVA",
            key="ask_image_button"
        ):

            if not image_question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                image_b64 = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                vision_messages = [
                    {
                        "role": "user",
                        "content": image_question
                    }
                ]

                with st.spinner(
                    "Analyzing image..."
                ):

                    answer, error = ask_ollama(
                        vision_messages,
                        model,
                        temperature,
                        image_base64=image_b64
                    )

                if error:

                    st.error(error)

                else:

                    st.markdown(answer)

        st.session_state[
            "image_question_mode"
        ] = None


# ============================================================
# VOICE WORKSPACE
# ============================================================

st.divider()

st.subheader("🎙️ Voice Assistant")

audio = st.audio_input(
    "Record your question"
)

if audio:

    st.audio(audio)

    audio_bytes = audio.getvalue()

    with st.spinner(
        "🎙️ Converting speech to text..."
    ):

        voice_text = speech_to_text(
            audio_bytes
        )

    if voice_text:

        st.session_state.voice_text = voice_text

        st.success(
            "Voice converted to text."
        )

        st.text_area(
            "Recognized speech",
            voice_text,
            height=100
        )

        if voice_text and not voice_text.startswith(
            ("I could not", "Speech recognition", "Voice processing")
        ):

            if st.button(
                "🚀 Send Voice Question"
            ):

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": voice_text
                    }
                )

                st.session_state.total_questions += 1

                st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for index, message in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Text-to-speech for assistant
        if message["role"] == "assistant":

            if st.button(
                "🔊 Read aloud",
                key=f"tts_{index}"
            ):

                with st.spinner(
                    "Generating voice..."
                ):

                    audio_file = text_to_speech(
                        message["content"]
                    )

                if audio_file:

                    st.audio(
                        audio_file,
                        format="audio/mp3"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message NOVA AI..."
)


# ============================================================
# CHAT PROCESSING
# ============================================================

if prompt:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.session_state.total_questions += 1

    with st.chat_message("user"):

        st.markdown(prompt)


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are NOVA AI, an intelligent personal AI assistant.

Your responsibilities:

1. Answer questions clearly and accurately.
2. Use Markdown formatting.
3. Explain difficult concepts simply.
4. Help with Python, SQL, Java, C, C++ and data analytics.
5. Help with Power BI, Tableau, Excel and machine learning.
6. Analyze uploaded document context when available.
7. Never claim that you saw information that is not provided.
8. If a document is supplied, prioritize its information.
9. Give complete code when the user asks for code.
10. Use headings, bullets and tables when useful.
11. Be concise for simple questions and detailed for complex questions.
"""

    file_context = build_file_context()

    system_message = {
        "role": "system",
        "content": system_prompt + file_context
    }


    # --------------------------------------------------------
    # API MESSAGES
    # --------------------------------------------------------

    api_messages = [
        system_message
    ]

    api_messages.extend(
        st.session_state.messages
    )


    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        placeholder = st.empty()

        with st.spinner(
            "NOVA is thinking..."
        ):

            answer, error = ask_ollama(
                api_messages,
                model,
                temperature
            )

        if error:

            placeholder.error(
                "❌ " + error
            )

        else:

            placeholder.markdown(
                answer
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    NOVA AI • AI Workspace • Chat • Vision • OCR • Voice • Documents
</div>
""",
    unsafe_allow_html=True
)
