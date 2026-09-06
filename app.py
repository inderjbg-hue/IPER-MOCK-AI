import os
import sys
import ssl
import tempfile
import time
import json
import re
import pandas as pd
import pypdf
import docx
import whisper
from groq import Groq
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------------------------
# 1. ENVIRONMENT & STREAMLIT CONFIGURATION
# ------------------------------------------------------------------------------
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" + os.pathsep + "/usr/local/bin" + os.pathsep + "/usr/bin"
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="IPER Placement & Interview Portal", layout="wide")

VIDEO_STORAGE_DIR = "saved_videos"
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. UI STYLING (NAVY BLUE & WHITE THEME)
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }

        .main .block-container {
            background-color: #FFFFFF !important;
            padding-top: 2rem !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }

        p, label, td, th {
            color: #1E293B !important;
            font-family: 'Inter', sans-serif !important;
        }

        div[role="radiogroup"] label {
            color: #0F172A !important;
        }
        
        div[role="radiogroup"] div[data-checked="true"] > div {
            background-color: #0F172A !important;
            border-color: #0F172A !important;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        div[data-baseweb="select"] > div,
        .stSelectbox select, 
        .stTextArea textarea, 
        .stTextInput input {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #94A3B8 !important;
            border-radius: 6px !important;
            font-family: 'Inter', sans-serif !important;
        }

        .stButton > button {
            background-color: #0F172A !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            border-radius: 6px !important;
            border: 1px solid #0F172A !important;
            padding: 0.5rem 1.4rem !important;
        }

        .stButton > button p {
            color: #FFFFFF !important;
        }

        .stButton > button:hover {
            background-color: #1E3A8A !important;
            border-color: #1E3A8A !important;
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed #0F172A !important;
            border-radius: 6px !important;
            padding: 12px !important;
            background-color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 6px !important;
            padding: 10px 14px !important;
            margin-bottom: 8px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            color: #0F172A !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            border-color: #0F172A !important;
            background-color: #F8FAFC !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & API INTEGRATION
# ------------------------------------------------------------------------------
@st.cache_resource
def load_speech_model():
    return whisper.load_model("base")

whisper_model = load_speech_model()

GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_API_KEY = None
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
elif os.getenv("GROQ_API_KEY"):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

# OpenAI is optional at startup. It is required only for the uploaded-GD
# speaker diarization/transcription workflow.
try:
    OPENAI_API_KEY = str(st.secrets.get("OPENAI_API_KEY", "") or "").strip()
except Exception:
    OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI is not None) else None

if "history" not in st.session_state:
    st.session_state["history"] = []

if "resume_details" not in st.session_state:
    st.session_state["resume_details"] = None

if "candidate_name" not in st.session_state:
    st.session_state["candidate_name"] = "Candidate"

def extract_text_from_file(file_obj):
    if file_obj is None:
        return ""
    
    filename = file_obj.name.lower()
    try:
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(file_obj)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            doc = docx.Document(file_obj)
            return "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            return f"[Uploaded Image File: {file_obj.name}]"
        else:
            return file_obj.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error extracting content from file: {str(e)}"

def _extract_speech_only_wav(audio_path):
    """Use aggressive WebRTC VAD to keep likely human-speech frames and reject music/noise."""
    if not WEBRTCVAD_AVAILABLE:
        return None, 0.0

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, 0.0

    pcm_cmd = [
        ffmpeg, "-y", "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"
    ]
    result = subprocess.run(pcm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not result.stdout:
        return None, 0.0

    raw = result.stdout
    sample_rate = 16000
    frame_ms = 30
    frame_bytes = int(sample_rate * frame_ms / 1000) * 2
    frame_count = len(raw) // frame_bytes
    if frame_count < 2:
        return None, 0.0

    vad = webrtcvad.Vad(3)
    speech_flags = []
    for i in range(frame_count):
        frame = raw[i * frame_bytes:(i + 1) * frame_bytes]
        try:
            speech_flags.append(vad.is_speech(frame, sample_rate))
        except Exception:
            speech_flags.append(False)

    # Require a meaningful amount of speech before sending audio to Whisper.
    speech_ratio = sum(speech_flags) / max(1, len(speech_flags))
    if sum(speech_flags) < 4 or speech_ratio < 0.015:
        return None, 0.0

    # Add ~300 ms padding around detected speech so words are not clipped.
    padding_frames = 10
    expanded = [False] * len(speech_flags)
    for i, is_speech in enumerate(speech_flags):
        if is_speech:
            start = max(0, i - padding_frames)
            end = min(len(expanded), i + padding_frames + 1)
            for j in range(start, end):
                expanded[j] = True

    speech_pcm = b"".join(
        raw[i * frame_bytes:(i + 1) * frame_bytes]
        for i, keep in enumerate(expanded) if keep
    )
    if len(speech_pcm) < frame_bytes * 4:
        return None, 0.0

    speech_duration = len(speech_pcm) / (sample_rate * 2)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_speech.wav")
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(speech_pcm)
        return tmp.name, speech_duration
    except Exception:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        return None, 0.0


def transcribe_indian_english_audio(audio_path):
    """Transcribe human speech only; aggressively suppress music/noise hallucinations."""
    prompt = "Official MBA placement interview response in Indian English. Transcribe only clearly audible human speech. Do not invent words from music, singing, background conversation, or noise."
    speech_path, speech_duration = _extract_speech_only_wav(audio_path)
    target_path = speech_path or audio_path
    try:
        result = whisper_model.transcribe(
            target_path,
            language="en",
            initial_prompt=prompt,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.72,
            logprob_threshold=-0.8,
            compression_ratio_threshold=2.2,
            fp16=False
        )

        # Keep only segments that Whisper itself considers speech-like.
        accepted = []
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            no_speech = float(seg.get("no_speech_prob", 1.0) or 1.0)
            avg_logprob = float(seg.get("avg_logprob", -99.0) or -99.0)
            if text and no_speech < 0.70 and avg_logprob > -1.05:
                accepted.append(text)

        transcript = " ".join(accepted).strip()
        # Guard against common Whisper hallucination loops.
        if transcript:
            words = re.findall(r"\b[\w']+\b", transcript.lower())
            if len(words) >= 8:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.25:
                    transcript = ""
        return transcript, speech_duration
    finally:
        if speech_path and os.path.exists(speech_path):
            try:
                os.remove(speech_path)
            except OSError:
                pass

STRICT_MENTOR_SYSTEM_PROMPT = """
You are a supportive MBA Placement Director and HR Reviewer at IPER Bhopal.
Address the candidate respectfully by name in all responses.
Provide clear, structured, professional guidance using user-friendly words.
"""

def get_groq_response(prompt):
    if not client:
        return "GROQ API Key is missing. Please add 'GROQ_API_KEY' in Streamlit App Settings -> Secrets."
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": STRICT_MENTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Execution Error: {str(e)}"

def render_video_recorder_component():
    """Browser recorder.

    Browsers such as Chrome commonly record WebM, while Safari may support MP4.
    The application converts any WebM uploaded from this recorder to MP4 server-side,
    so the saved interview file is always MP4.
    """
    html_code = """
    <div style="font-family: 'Inter', sans-serif; border: 1px solid #CBD5E1; border-radius: 6px; padding: 15px; background: #FFFFFF;">
        <video id="preview" autoplay playsinline muted style="width: 100%; max-height: 240px; background: #000; border-radius: 4px;"></video>
        <div style="margin-top: 12px; display: flex; gap: 10px; flex-wrap: wrap;">
            <button id="startBtn" onclick="startRec()" style="background: #0F172A; color: white; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: 600; font-family: 'Inter', sans-serif;">Start Video Recording</button>
            <button id="stopBtn" onclick="stopRec()" disabled style="background: #475569; color: white; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: 600; font-family: 'Inter', sans-serif;">Stop & Save</button>
            <a id="downloadAnchor" style="display:none; background: #0F172A; color: white; text-decoration: none; padding: 8px 14px; border-radius: 4px; font-size: 13px; font-weight: 600; font-family: 'Inter', sans-serif;">Download Recording</a>
        </div>
        <div id="status" style="margin-top: 10px; font-size: 13px; color: #0F172A; font-weight: 500;">Status: Camera Ready</div>
        <div style="margin-top: 6px; font-size: 12px; color: #64748B;">The portal converts WebM recordings to MP4 automatically when you upload them below.</div>
    </div>
    <script>
        let recorder, chunks = [], streamRef, selectedMime = 'video/webm';
        async function startRec() {
            try {
                chunks = [];
                document.getElementById('downloadAnchor').style.display = 'none';
                streamRef = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
                document.getElementById('preview').srcObject = streamRef;

                const mimeCandidates = [
                    'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
                    'video/mp4',
                    'video/webm;codecs=vp9,opus',
                    'video/webm;codecs=vp8,opus',
                    'video/webm'
                ];
                selectedMime = mimeCandidates.find(m => MediaRecorder.isTypeSupported(m)) || '';
                if (!selectedMime) throw new Error('This browser does not support video recording.');

                recorder = new MediaRecorder(streamRef, { mimeType: selectedMime });
                recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
                recorder.onstop = () => {
                    const isMp4 = selectedMime.toLowerCase().includes('mp4');
                    const extension = isMp4 ? 'mp4' : 'webm';
                    const blob = new Blob(chunks, { type: selectedMime.split(';')[0] });
                    const downloadBtn = document.getElementById('downloadAnchor');
                    downloadBtn.href = URL.createObjectURL(blob);
                    downloadBtn.download = 'iper_interview_' + Date.now() + '.' + extension;
                    downloadBtn.textContent = isMp4 ? 'Download MP4 Recording' : 'Download Recording (WebM → MP4 in Portal)';
                    downloadBtn.style.display = 'inline-block';
                };
                recorder.start(1000);
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('status').innerText = 'Status: Recording in Progress...';
            } catch (err) {
                document.getElementById('status').innerText = 'Camera Error: ' + err.message;
                if (streamRef) streamRef.getTracks().forEach(track => track.stop());
            }
        }
        function stopRec() {
            if (recorder && recorder.state !== 'inactive') recorder.stop();
            if(streamRef) streamRef.getTracks().forEach(track => track.stop());
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').innerText = 'Status: Recording completed.';
        }
    </script>
    """
    components.html(html_code, height=365)


def convert_video_to_mp4(input_path):
    """Convert a video to browser-compatible H.264/AAC MP4 using FFmpeg."""
    input_path = str(input_path)
    output_path = str(Path(input_path).with_suffix('.mp4'))
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg is not installed. Add ffmpeg to packages.txt and redeploy.')
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        return output_path
    cmd = [
        ffmpeg, '-y', '-i', input_path,
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart', output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError('FFmpeg could not convert the recording to MP4.')
    return output_path


def get_media_duration_seconds(media_path):
    """Return media duration in seconds using ffprobe/ffmpeg."""
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(media_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return max(0.0, float(result.stdout.strip()))
    except (TypeError, ValueError):
        return 0.0


def communication_metrics(transcript, duration_seconds=0.0):
    """Deterministic speech metrics plus filler-word detection for AI feedback."""
    text = (transcript or '').strip()
    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)
    duration = float(duration_seconds or 0)
    wpm = round((word_count / duration) * 60, 1) if duration > 0 else 0.0

    filler_patterns = {
        'um': r'\bum\b', 'uh': r'\buh\b', 'er': r'\ber\b',
        'you know': r'\byou know\b', 'like': r'\blike\b',
        'basically': r'\bbasically\b', 'actually': r'\bactually\b',
        'I mean': r'\bi mean\b', 'kind of': r'\bkind of\b',
        'sort of': r'\bsort of\b', 'right': r'\bright\b'
    }
    filler_counts = {name: len(re.findall(pattern, text, flags=re.I)) for name, pattern in filler_patterns.items()}
    filler_counts = {k: v for k, v in filler_counts.items() if v > 0}
    filler_total = sum(filler_counts.values())
    filler_rate = round((filler_total / word_count) * 100, 1) if word_count else 0.0
    return {
        'word_count': word_count,
        'duration_seconds': round(duration, 1),
        'duration_minutes': round(duration / 60, 2),
        'wpm': wpm,
        'filler_total': filler_total,
        'filler_rate': filler_rate,
        'filler_counts': filler_counts
    }

# ------------------------------------------------------------------------------
# 4. QUESTION REPOSITORY
# ------------------------------------------------------------------------------
EXHAUSTIVE_QUESTIONS = {
    "General and Core Skills": [
        "1. Walk me through your resume highlighting your core achievements.",
        "2. What are your top 3 professional strengths and 2 areas of growth?",
        "3. Describe a situation where you led a team under a strict deadline.",
        "4. How do you handle constructive feedback from team managers?",
        "5. What is your 5-year career blueprint after completing your MBA?",
        "6. Tell me about a time you resolved a difficult conflict within a group.",
        "7. How do you prioritize tasks when managing multiple project deadlines?",
        "8. Describe a major failure or setback you experienced and how you bounced back.",
        "9. Why did you choose IPER for your management degree?",
        "10. How do you stay updated with current business and economic trends?",
        "11. Give an example of how you used data to solve a complex problem.",
        "12. What makes you a suitable candidate for corporate management roles?",
        "13. How do you adapt when project goals change unexpectedly?",
        "14. Describe your personal leadership and communication style.",
        "15. What motivates you to perform at your best in a workplace setting?",
        "16. Tell me something about Yourself."
    ],
    "Marketing": [
        "1. Differentiate between push and pull marketing strategies with real examples.",
        "2. How do you design a high-converting digital marketing funnel?",
        "3. Explain the 7 Ps of Service Marketing in the context of retail.",
        "4. How do you measure Customer Lifetime Value (CLV) and Customer Acquisition Cost (CAC)?",
        "5. What is the difference between B2B and B2C marketing approaches?",
        "6. Explain the concept of Brand Positioning and how to create a Unique Selling Proposition (USP).",
        "7. How would you handle a PR crisis for a consumer products brand?",
        "8. Explain Search Engine Optimization (SEO) vs. Search Engine Marketing (SEM).",
        "9. What is Market Segmentation, and how do Demographics differ from Psychographics?",
        "10. Describe the steps involved in launching a new product in a competitive market.",
        "11. How does Content Marketing build long-term customer loyalty?",
        "12. What key metrics do you track to measure social media campaign success?",
        "13. Explain the Product Life Cycle (PLC) and marketing actions at each stage.",
        "14. How do influencer marketing strategies impact consumer purchasing behavior?",
        "15. What is Guerrilla Marketing, and when is it most effective?"
    ],
    "Finance": [
        "1. Explain the 3 main financial statements and how they link together.",
        "2. What is Working Capital, and how do you calculate Net Working Capital?",
        "3. Describe the Discounted Cash Flow (DCF) valuation methodology.",
        "4. What is the difference between Capital Expenditure (CapEx) and Operational Expenditure (OpEx)?",
        "5. Explain the concept of the Time Value of Money (TVM).",
        "6. What is the Capital Asset Pricing Model (CAPM) and how is Beta used?",
        "7. Differentiate between Debt Financing and Equity Financing.",
        "8. What is NPV (Net Present Value) and IRR (Internal Rate of Return)?",
        "9. Explain the Liquidity Ratios vs. Profitability Ratios with formulas.",
        "10. What is a Balance Sheet, and why must assets always equal liabilities plus equity?",
        "11. How does inflation impact interest rates and corporate cash flows?",
        "12. What is Financial Leverage, and how does it affect risk and return?",
        "13. Explain the difference between primary and secondary financial markets.",
        "14. How do you analyze a company's financial health using a Cash Flow Statement?",
        "15. What is Dupont Analysis, and what are its three key components?"
    ],
    "Human Resource (HR)": [
        "1. Explain the step-by-step recruitment and selection workflow.",
        "2. How do you handle workplace conflict between two senior employees?",
        "3. Explain the 360-Degree Performance Appraisal technique.",
        "4. What is the difference between Training and Development?",
        "5. How do you design a competitive Compensation and Benefits structure?",
        "6. What strategies would you use to reduce employee turnover in a high-stress sector?",
        "7. Explain HR Analytics and how data improves talent management decisions.",
        "8. What is the role of HR during corporate restructuring or mergers?",
        "9. How do you foster Diversity, Equity, and Inclusion (DEI) in an organization?",
        "10. Explain the concept of Organizational Culture and how HR influences it.",
        "11. What is the difference between Job Description and Job Specification?",
        "12. How do you conduct an effective Exit Interview?",
        "13. What is Succession Planning, and why is it critical for business continuity?",
        "14. Explain key labor laws and compliance regulations every HR manager should know.",
        "15. How do you manage performance issues using Performance Improvement Plans (PIP)?"
    ],
    "Banking and Finance": [
        "1. Explain the difference between Retail Banking and Corporate Banking.",
        "2. What are Non-Performing Assets (NPAs), and how do banks manage them?",
        "3. Explain the role and functions of the Reserve Bank of India (RBI).",
        "4. What is repo rate and reverse repo rate, and how do they impact the economy?",
        "5. Describe the KYC (Know Your Customer) guidelines and their importance.",
        "6. What is the difference between a Savings Account and a Current Account?",
        "7. Explain the concept of Credit Risk and how credit scores are calculated.",
        "8. What is the Basel III framework, and why is capital adequacy important?",
        "9. How do commercial banks generate profit?",
        "10. What is the difference between NEFT, RTGS, and IMPS payment systems?",
        "11. What is an Initial Public Offering (IPO), and how is it underwritten?",
        "12. Explain Asset Reconstruction Companies (ARCs) and their role in resolving NPAs.",
        "13. What is the difference between Money Market and Capital Market?",
        "14. How do digital banking platforms impact traditional brick-and-mortar operations?",
        "15. Explain the concept of Microfinance and its social and economic impact."
    ],
    "Tourism and Services Industry": [
        "1. What are the unique characteristics of services marketing compared to physical goods?",
        "2. How do you manage service quality expectations using the SERVQUAL model?",
        "3. Explain the concept of Yield Management in the hotel and airline sectors.",
        "4. How do you handle an unhappy guest complaint in hospitality management?",
        "5. What is Sustainable Tourism, and how can companies promote eco-friendly travel?",
        "6. Explain the role of Destination Marketing Organizations (DMOs).",
        "7. How does customer relationship management (CRM) build customer loyalty in services?",
        "8. Describe the impact of online travel aggregators (OTAs) on traditional agencies.",
        "9. What is Moment of Truth in service delivery, and why is it vital?",
        "10. How do service blueprints help streamline operational workflows?",
        "11. Explain crisis management strategies for hospitality and tourism businesses.",
        "12. How has technology transformed guest check-in and booking experiences?",
        "13. What is the economic multiplier effect in international tourism?",
        "14. How do seasonal demand fluctuations affect staff planning in travel and tourism?",
        "15. Explain Event Management fundamentals when organizing large corporate conferences."
    ]
}

SPECIALIZATIONS = ["Marketing", "Finance", "Human Resource (HR)", "Banking and Finance", "Tourism and Services Industry"]
IPER_RECRUITERS = ["Amul", "Asian Paints", "HDFC Bank", "ICICI Securities", "Deloitte", "Trident Group", "Berger Paints"]

CORE_INTERVIEW_QUESTIONS = [
    "Tell me something about yourself.", "Walk me through your resume.", "Why did you choose MBA?",
    "Why did you choose your specialization?", "What have you learned during your MBA?",
    "Which subject do you like the most and why?", "What did you learn from a curricular activity?",
    "What did you learn from an extra-curricular activity?", "What did you learn from your internship or live project?",
    "Why should we hire you?", "What are your strengths?", "What is one area you are working to improve?",
    "Tell me about a time you worked in a team.", "Tell me about a time you handled a difficult situation.",
    "Where do you see yourself in five years?", "Why do you want to join our company?",
    "What do you know about our company?", "What are your career goals after MBA?", "Do you have any questions for us?"
]
CURRICULAR_ACTIVITIES = ["Case Study", "Group Project", "Presentation", "Research Project", "Business Simulation", "Internship", "Live Project", "Classroom Activity"]
EXTRA_CURRICULAR_ACTIVITIES = ["Sports", "Cultural Event", "Club Activity", "Volunteering", "Event Management", "Competition", "Student Leadership", "Community Activity"]

# ------------------------------------------------------------------------------
# 5. STUDENT AUTHENTICATION & ACCOUNT MANAGEMENT
# ------------------------------------------------------------------------------
# IMPORTANT:
# The original version used a relative SQLite database ("students.db"). That
# database is NOT guaranteed to survive a restart/rebuild on hosted platforms.
# This version supports a persistent PostgreSQL DATABASE_URL (recommended for
# Streamlit Cloud) while retaining SQLite as a local-development fallback.

import sqlite3
import hashlib
import secrets
import shutil
from datetime import datetime
import base64
import hmac
import urllib.parse
import re
import subprocess
import wave
from pathlib import Path

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    WEBRTCVAD_AVAILABLE = False

try:
    import psycopg2
    from psycopg2 import IntegrityError as PostgresIntegrityError
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None
    PostgresIntegrityError = Exception
    POSTGRES_AVAILABLE = False

DATABASE_PATH = os.getenv("DATABASE_PATH", "students.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL and "DATABASE_URL" in st.secrets:
    DATABASE_URL = str(st.secrets["DATABASE_URL"]).strip()

ALLOWED_EMAIL_DOMAIN = "@iper.ac.in"
USING_POSTGRES = bool(DATABASE_URL)


class HybridRow(dict):
    """Dict-like DB row that also supports integer indexing used by legacy code."""
    def __init__(self, columns, values):
        super().__init__(zip(columns, values))
        self._values = list(values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


class PostgresCursorWrapper:
    def __init__(self, cursor):
        self.cursor = cursor

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is None:
            return None
        return HybridRow([d.name for d in self.cursor.description], row)

    def fetchall(self):
        rows = self.cursor.fetchall()
        if not rows:
            return []
        columns = [d.name for d in self.cursor.description]
        return [HybridRow(columns, row) for row in rows]


class PostgresConnectionWrapper:
    """Small compatibility layer so the existing SQLite-style SQL can run on PostgreSQL."""
    def __init__(self, url):
        self.conn = psycopg2.connect(url, connect_timeout=15)
        self.conn.autocommit = False

    @staticmethod
    def _translate_sql(sql):
        # SQLite uses ? placeholders; psycopg2 uses %s.
        sql = sql.replace("?", "%s")
        sql = re.sub(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", "INSERT INTO", sql, flags=re.I)
        sql = re.sub(r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b", "BIGSERIAL PRIMARY KEY", sql, flags=re.I)
        return sql

    def execute(self, sql, params=None):
        # PostgreSQL equivalent of the SQLite migration query.
        if sql.strip().lower().startswith("pragma table_info(gd_rooms)"):
            cur = self.conn.cursor()
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'gd_rooms'
                ORDER BY ordinal_position
            """)
            return PostgresCursorWrapper(cur)

        translated = self._translate_sql(sql)
        # The only INSERT OR IGNORE in the app is safe to express as ON CONFLICT DO NOTHING.
        if translated.lstrip().upper().startswith("INSERT INTO GD_ROOMS") and "ON CONFLICT" not in translated.upper():
            translated = translated.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

        cur = self.conn.cursor()
        cur.execute(translated, params or ())
        return PostgresCursorWrapper(cur)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


def is_valid_iper_email(email):
    email = email.strip().lower()
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@iper\.ac\.in", email))


def get_db_connection():
    """Use persistent PostgreSQL when DATABASE_URL is configured; SQLite locally otherwise."""
    if DATABASE_URL:
        if not POSTGRES_AVAILABLE:
            raise RuntimeError(
                "DATABASE_URL is configured, but psycopg2-binary is not installed. "
                "Add psycopg2-binary to requirements.txt and redeploy."
            )
        return PostgresConnectionWrapper(DATABASE_URL)

    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                scholar_id TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interview_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                domain TEXT,
                score INTEGER,
                mode TEXT,
                question TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gd_rooms (
                slot INTEGER PRIMARY KEY,
                code TEXT UNIQUE,
                topic TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open',
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                recording_status TEXT DEFAULT 'Not Started'
            )
        """)

        # Backward-compatible migration: remember which student created/hosts the GD.
        existing_columns = [row[0] if USING_POSTGRES else row[1]
                            for row in conn.execute("PRAGMA table_info(gd_rooms)").fetchall()]
        if "host_scholar_id" not in existing_columns:
            conn.execute("ALTER TABLE gd_rooms ADD COLUMN host_scholar_id TEXT")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS gd_participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER NOT NULL,
                scholar_id TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                joined_at TEXT NOT NULL,
                left_at TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                UNIQUE(slot, scholar_id),
                FOREIGN KEY(slot) REFERENCES gd_rooms(slot)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gd_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot INTEGER NOT NULL,
                scholar_id TEXT NOT NULL,
                voice_score INTEGER NOT NULL,
                perspective_score INTEGER NOT NULL,
                participation_score INTEGER NOT NULL,
                camera_clarity_score INTEGER NOT NULL,
                strengths TEXT,
                improvements TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gd_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scholar_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                participant_names TEXT,
                transcript_json TEXT,
                report_json TEXT,
                video_filename TEXT,
                created_at TEXT NOT NULL
            )
        """)

        for slot in range(1, 8):
            conn.execute(
                "INSERT OR IGNORE INTO gd_rooms (slot, code, topic, status, created_at) VALUES (?, NULL, ?, 'Open', ?)",
                (slot, 'Not allocated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    ).hex()
    return password_hash, salt


def verify_password(password, stored_hash, stored_salt):
    password_hash, _ = hash_password(password, stored_salt)
    return secrets.compare_digest(password_hash, stored_hash)


def create_student(first_name, last_name, scholar_id, email, password):
    email = email.strip().lower()
    scholar_id = scholar_id.strip().upper()
    password_hash, password_salt = hash_password(password)

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO students
            (first_name, last_name, scholar_id, email, password_hash, password_salt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_name.strip(), last_name.strip(), scholar_id, email,
                password_hash, password_salt,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return True, "Account created successfully. You can now sign in."
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        message = str(exc).lower()
        if "unique" in message or "duplicate" in message:
            if "email" in message:
                return False, "This email ID is already registered. Please sign in."
            if "scholar_id" in message:
                return False, "This Scholar ID is already registered."
            return False, "This account could not be created because the details already exist."
        raise
    finally:
        conn.close()


def authenticate_student(email, password):
    email = email.strip().lower()
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE email = ?", (email,)
        ).fetchone()
    finally:
        conn.close()

    if row and verify_password(password, row["password_hash"], row["password_salt"]):
        return dict(row)
    return None


def reset_student_password(email, scholar_id, new_password):
    """Reset a student's password after verifying their official email and Scholar ID."""
    email = email.strip().lower()
    scholar_id = scholar_id.strip().upper()
    password_hash, password_salt = hash_password(new_password)

    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id FROM students WHERE email = ? AND scholar_id = ?",
            (email, scholar_id),
        ).fetchone()

        if not row:
            return False, "We could not verify those details. Please check your official email ID and Scholar ID."

        conn.execute(
            "UPDATE students SET password_hash = ?, password_salt = ? WHERE id = ?",
            (password_hash, password_salt, row["id"]),
        )
        conn.commit()
        return True, "Password reset successfully. You can now sign in with your new password."
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def load_student_attempts(student_id):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT timestamp, domain, score, mode, question
            FROM interview_attempts
            WHERE student_id = ?
            ORDER BY id ASC
            """,
            (student_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "Timestamp": row["timestamp"], "Candidate": "Student",
            "Domain": row["domain"] or "", "Score": row["score"] or 0,
            "Mode": row["mode"] or "", "Question": row["question"] or "",
        }
        for row in rows
    ]


def save_student_attempt(student_id, domain, score, mode, question):
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO interview_attempts
            (student_id, timestamp, domain, score, mode, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                domain, int(score), mode, question,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# Show a clear deployment warning rather than silently using an ephemeral DB.
def database_status_message():
    if DATABASE_URL:
        return "Persistent student database: PostgreSQL"
    return "Local student database: SQLite (use DATABASE_URL in production for persistent accounts)"


def migrate_local_sqlite_to_postgres():
    """One-time best-effort migration of an existing local students.db into PostgreSQL.

    This runs only when DATABASE_URL is configured, the local SQLite file exists,
    and the remote students table is empty. Password hashes/salts are copied as-is,
    so existing student passwords continue to work.
    """
    if not DATABASE_URL or not os.path.exists(DATABASE_PATH):
        return

    local = sqlite3.connect(DATABASE_PATH)
    local.row_factory = sqlite3.Row
    remote = get_db_connection()
    try:
        remote_count = remote.execute("SELECT COUNT(*) AS n FROM students").fetchone()["n"]
        if int(remote_count or 0) > 0:
            return

        tables = ["students", "interview_attempts", "gd_rooms", "gd_participants", "gd_feedback"]
        for table in tables:
            rows = local.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            columns = rows[0].keys()
            placeholders = ", ".join(["?"] * len(columns))
            column_sql = ", ".join(columns)
            for row in rows:
                remote.execute(
                    f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                    tuple(row[c] for c in columns),
                )

        # The imported SQLite IDs are explicit, so advance PostgreSQL sequences
        # before any new accounts/attempts are created.
        for table in ["students", "interview_attempts", "gd_participants", "gd_feedback"]:
            remote.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), COALESCE((SELECT MAX(id) FROM {table}), 1), true)"
            )
        remote.commit()
    except Exception:
        try:
            remote.rollback()
        except Exception:
            pass
        # Never prevent the app from starting just because an optional migration failed.
    finally:
        local.close()
        remote.close()



GD_MAX_PARTICIPANTS = 7
GD_MAX_MINUTES = 10
GD_ROOM_COUNT = 7

# Optional external video engine configuration for in-built GD practice.
# The app gracefully falls back to the local/in-browser GD flow when these are not configured.
VIDEO_ENGINE_URL = os.getenv("VIDEO_ENGINE_URL", st.secrets.get("VIDEO_ENGINE_URL", "")).strip().rstrip("/")
VIDEO_ENGINE_JWT_SECRET = os.getenv("VIDEO_ENGINE_JWT_SECRET", st.secrets.get("VIDEO_ENGINE_JWT_SECRET", "")).strip()

GD_TOPICS = [
    "Artificial Intelligence: Job Creator or Job Killer?",
    "Should college education be skill-based rather than degree-based?",
    "Work From Home vs Work From Office",
    "Is Social Media a Boon or a Curse?",
    "Should AI be regulated?",
    "Digital Payments and the Future of Cash",
    "Startup Culture vs Stable Corporate Jobs",
    "Is India ready for a cashless economy?",
    "Sustainability vs Profitability",
    "Can India become a global manufacturing hub?",
    "Online Education vs Classroom Education",
    "Is influencer marketing trustworthy?",
    "Electric Vehicles: Future or Fad?",
    "Should internships be mandatory for every student?",
    "Data Privacy in the Digital Age",
    "Is competition good for students?",
    "Work-Life Balance vs Career Growth",
    "Should companies hire for skills rather than degrees?",
    "Is failure necessary for success?",
    "Leadership: Born or Made?",
    "Is customer experience more important than product quality?",
    "Can technology replace human creativity?",
    "Should college attendance be compulsory?",
    "Are advertisements influencing consumers too much?",
    "Ethical Issues in Artificial Intelligence",
    "Should businesses take political stands?",
    "Is entrepreneurship for everyone?",
    "Green Marketing: Genuine Need or Branding Strategy?",
    "Should companies adopt a four-day work week?",
    "Gig Economy: Opportunity or Exploitation?",
    "Is remote work reducing organizational culture?",
    "Should employees be allowed to work anywhere?",
    "Can India lead the global AI revolution?",
    "Technology and Human Relationships",
    "Should social media platforms be responsible for misinformation?",
    "Is economic growth possible without environmental damage?",
    "Brand Loyalty in the Age of Online Shopping",
    "Should financial literacy be compulsory in colleges?",
    "Is cryptocurrency the future of money?",
    "Digital Banking vs Traditional Banking",
    "Should companies monitor employee productivity digitally?",
    "Is diversity important for business success?",
    "Can India achieve sustainable development?",
    "Should exams be replaced by continuous assessment?",
    "The Future of the Indian Retail Industry",
    "Does advertising create artificial needs?",
    "Should businesses prioritize local suppliers?",
    "Is globalization good for developing countries?",
    "Should college students be allowed to use AI for assignments?",
    "AI in Recruitment: Fairness vs Efficiency",
    "Can emotional intelligence be more important than IQ at work?",
    "Should companies disclose their salary ranges?",
    "Performance Pay vs Fixed Salary",
    "Is job security becoming less important?",
    "Should organizations prioritize employee wellbeing?",
    "Is customer data the new oil?",
    "Can digital marketing replace traditional marketing?",
    "Are discounts destroying brand value?",
    "Should luxury brands embrace mass-market collaborations?",
    "India's youth and entrepreneurship",
    "Is a high salary the best measure of career success?",
    "Should students pursue passion or job security?",
    "Is networking more important than academic performance?",
    "Should companies invest more in employee training?",
    "Can automation improve workplace productivity?",
    "Should managers use AI to evaluate employees?",
    "Is hybrid work the best future of work?",
    "Should companies have unlimited leave policies?",
    "Is employee loyalty still relevant?",
    "Can small businesses compete with e-commerce giants?",
    "Is quick commerce changing consumer behavior permanently?",
    "Should India prioritize domestic consumption?",
    "Tourism as a driver of economic development",
    "Is sustainable tourism practical?",
    "Should public transport be free in major cities?",
    "Electric public transport and urban mobility",
    "Should cities discourage private vehicles?",
    "Is population growth an economic advantage or challenge?",
    "Should businesses be responsible for social development?",
    "Corporate Social Responsibility: Responsibility or Marketing?",
    "Should profit be the primary goal of business?",
    "Can ethical business practices create competitive advantage?",
    "Is brand reputation more valuable than short-term sales?",
    "Should CEOs be active on social media?",
    "The impact of short-form video on attention spans",
    "Is digital detox necessary?",
    "Should children have restricted social media access?",
    "Online privacy vs national security",
    "Should AI-generated content be labelled?",
    "Is misinformation a bigger threat than fake news?",
    "Should voting be compulsory?",
    "Youth participation in nation building",
    "Can sports create stronger communities?",
    "Should colleges focus more on employability?",
    "Is academic pressure helping or harming students?",
    "Mental resilience in professional life",
    "Should companies value soft skills equally with technical skills?",
    "The importance of communication skills in management",
    "Is multitasking reducing productivity?",
    "Should employees be allowed to disconnect after work?",
    "Leadership lessons from Indian businesses",
    "Future of management education in India",
    "Can India become a knowledge economy?",
    "The role of women in India's workforce",
    "Should organizations prioritize gender diversity?",
    "Is meritocracy possible without equal opportunity?",
    "Should companies recruit directly from colleges?",
    "Campus placements vs independent job search",
    "Are internships becoming more important than degrees?",
]

GD_DOS = [
    "Understand the topic before speaking.",
    "Open with a clear and relevant point when appropriate.",
    "Listen actively and build on other participants' ideas.",
    "Use facts, examples and business or real-world context.",
    "Keep your contribution concise and structured.",
    "Invite quieter members into the discussion.",
    "Disagree with ideas respectfully, not with people.",
    "Help the group move toward a balanced conclusion."
]

GD_DONTS = [
    "Do not interrupt repeatedly.",
    "Do not dominate the discussion.",
    "Do not attack or ridicule another participant.",
    "Do not invent statistics to sound convincing.",
    "Do not repeat the same point without adding value.",
    "Do not turn the GD into a one-to-one argument.",
    "Do not stay silent for the entire discussion.",
    "Do not force a conclusion without listening to the group."
]


def generate_gd_code():
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        code = "".join(secrets.choice(alphabet) for _ in range(6))
        conn = get_db_connection()
        exists = conn.execute("SELECT 1 FROM gd_rooms WHERE code = ?", (code,)).fetchone()
        conn.close()
        if not exists:
            return code


def reset_gd_slot(slot, topic):
    code = generate_gd_code()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    conn.execute("DELETE FROM gd_participants WHERE slot = ?", (slot,))
    conn.execute("""
        UPDATE gd_rooms
        SET code=?, topic=?, status='Open', created_at=?, started_at=NULL, ended_at=NULL, recording_status='Not Started'
        WHERE slot=?
    """, (code, topic, now, slot))
    conn.commit()
    conn.close()
    return code


def create_student_gd_session(topic, host_student):
    """Create/reuse the single student-driven GD room (slot 1)."""
    code = generate_gd_code()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        # A new session replaces the previous student GD session.
        conn.execute("DELETE FROM gd_participants WHERE slot = 1")
        conn.execute("""
            UPDATE gd_rooms
            SET code=?, topic=?, status='Open', created_at=?, started_at=NULL,
                ended_at=NULL, recording_status='Not Started', host_scholar_id=?
            WHERE slot=1
        """, (code, topic.strip(), now, host_student["scholar_id"]))
        conn.commit()
        return code
    finally:
        conn.close()


def get_student_gd_room():
    rooms = get_gd_rooms()
    return next((r for r in rooms if r.get("slot") == 1), None)


def start_student_gd_session(code, scholar_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        room = conn.execute(
            "SELECT * FROM gd_rooms WHERE slot=1 AND code=?", (code.strip().upper(),)
        ).fetchone()
        if not room:
            return False, "Invalid GD room code."
        if room["host_scholar_id"] != scholar_id:
            return False, "Only the student who generated the GD code can start this session."
        if room["status"] == "Ended":
            return False, "This GD session has already ended. Create a new session."
        conn.execute(
            "UPDATE gd_rooms SET status='Active', started_at=?, recording_status='Starting' WHERE slot=1",
            (now,)
        )
        conn.commit()
        return True, "GD session started. Recording will begin when the host enters the room."
    finally:
        conn.close()


def end_student_gd_session(code, scholar_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        room = conn.execute(
            "SELECT * FROM gd_rooms WHERE slot=1 AND code=?", (code.strip().upper(),)
        ).fetchone()
        if not room:
            return False, "GD room not found."
        if room["host_scholar_id"] != scholar_id:
            return False, "Only the student who created the GD can end the session."
        conn.execute(
            "UPDATE gd_rooms SET status='Ended', ended_at=?, recording_status='Ended' WHERE slot=1",
            (now,)
        )
        conn.commit()
        return True, "GD session ended."
    finally:
        conn.close()


def get_gd_rooms():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM gd_rooms ORDER BY slot").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_gd_participants(slot):
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT scholar_id, first_name, last_name, joined_at
        FROM gd_participants
        WHERE slot=? AND active=1
        ORDER BY joined_at
    """, (slot,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def join_gd_room(slot, code, student):
    conn = get_db_connection()
    try:
        room = conn.execute("SELECT * FROM gd_rooms WHERE slot=? AND code=?", (slot, code.strip().upper())).fetchone()
        if not room:
            return False, "Invalid GD room code for this room."
        if room["status"] == "Ended":
            return False, "This GD has already ended."
        active_count = conn.execute("SELECT COUNT(*) FROM gd_participants WHERE slot=? AND active=1", (slot,)).fetchone()[0]
        existing = conn.execute("SELECT * FROM gd_participants WHERE slot=? AND scholar_id=?", (slot, student["scholar_id"])).fetchone()
        if existing and existing["active"]:
            return True, "You are already registered in this GD room."
        if active_count >= GD_MAX_PARTICIPANTS:
            return False, f"This room is full. Maximum {GD_MAX_PARTICIPANTS} students are allowed."
        conn.execute("UPDATE gd_participants SET active=0, left_at=? WHERE scholar_id=? AND active=1",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), student["scholar_id"]))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if existing:
            conn.execute("UPDATE gd_participants SET first_name=?, last_name=?, joined_at=?, left_at=NULL, active=1 WHERE id=?",
                         (student["first_name"], student["last_name"], now, existing["id"]))
        else:
            conn.execute("INSERT INTO gd_participants(slot, scholar_id, first_name, last_name, joined_at, active) VALUES (?, ?, ?, ?, ?, 1)",
                         (slot, student["scholar_id"], student["first_name"], student["last_name"], now))
        conn.commit()
        return True, "You have joined the GD room."
    finally:
        conn.close()


def leave_gd_room(slot, scholar_id):
    conn = get_db_connection()
    conn.execute("UPDATE gd_participants SET active=0, left_at=? WHERE slot=? AND scholar_id=?",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), slot, scholar_id))
    conn.commit()
    conn.close()


def set_gd_status(slot, status, recording_status=None):
    conn = get_db_connection()
    if status == "Active":
        conn.execute("UPDATE gd_rooms SET status='Active', started_at=? WHERE slot=?",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), slot))
    elif status == "Ended":
        conn.execute("UPDATE gd_rooms SET status='Ended', ended_at=?, recording_status=COALESCE(?, recording_status) WHERE slot=?",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), recording_status, slot))
    else:
        conn.execute("UPDATE gd_rooms SET status=?, recording_status=COALESCE(?, recording_status) WHERE slot=?",
                     (status, recording_status, slot))
    conn.commit()
    conn.close()


def save_gd_feedback(slot, student, voice, perspective, participation, camera, strengths, improvements):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO gd_feedback
        (slot, scholar_id, voice_score, perspective_score, participation_score, camera_clarity_score, strengths, improvements, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (slot, student["scholar_id"], voice, perspective, participation, camera,
          strengths.strip(), improvements.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_student_gd_feedback(scholar_id):
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT * FROM gd_feedback WHERE scholar_id=? ORDER BY id DESC
    """, (scholar_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]



def _ffmpeg_extract_gd_audio(video_path):
    """Convert uploaded GD video to compact mono MP3 for diarized transcription."""
    out = tempfile.NamedTemporaryFile(delete=False, suffix="_gd_audio.mp3")
    out.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-b:a", "64k",
        out.name
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        if result.returncode != 0 or not os.path.exists(out.name) or os.path.getsize(out.name) == 0:
            try: os.remove(out.name)
            except OSError: pass
            return None
        return out.name
    except Exception:
        try: os.remove(out.name)
        except OSError: pass
        return None


def transcribe_gd_with_diarization(audio_path):
    """Use OpenAI's speaker-diarized transcription when configured."""
    if not openai_client:
        return None, "OPENAI_API_KEY is not configured. Add it to Streamlit Secrets for multi-speaker GD analysis."
    try:
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if size_mb > 24.5:
            return None, "The extracted GD audio is still above the 25 MB transcription limit."
        with open(audio_path, "rb") as audio_file:
            result = openai_client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto"
            )
        raw = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        segments = raw.get("segments") or []
        clean = []
        for seg in segments:
            speaker = str(seg.get("speaker") or "speaker_0")
            phrase = str(seg.get("text") or "").strip()
            if not phrase:
                continue
            clean.append({
                "speaker": speaker,
                "start": round(float(seg.get("start") or 0), 2),
                "end": round(float(seg.get("end") or 0), 2),
                "text": phrase
            })
        if not clean:
            return None, "No clear human speech was detected in the GD recording."
        return clean, None
    except Exception as exc:
        return None, f"GD transcription failed: {exc}"


def _format_gd_transcript(segments, speaker_map=None):
    speaker_map = speaker_map or {}
    lines = []
    for seg in segments:
        speaker = speaker_map.get(seg["speaker"], seg["speaker"])
        lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}s] {speaker}: {seg['text']}")
    return "\n".join(lines)


def _gd_speaker_stats(segments, speaker_map=None):
    speaker_map = speaker_map or {}
    stats = {}
    for seg in segments:
        key = speaker_map.get(seg["speaker"], seg["speaker"])
        words = re.findall(r"\b[\w']+\b", seg["text"])
        duration = max(0.0, seg["end"] - seg["start"])
        entry = stats.setdefault(key, {"speaking_seconds": 0.0, "words": 0, "turns": 0, "filler_words": 0})
        entry["speaking_seconds"] += duration
        entry["words"] += len(words)
        entry["turns"] += 1
        filler_pattern = r"\b(um+|uh+|er+|you know|like|basically|actually|i mean|kind of|sort of)\b"
        entry["filler_words"] += len(re.findall(filler_pattern, seg["text"], flags=re.I))
    total = sum(v["speaking_seconds"] for v in stats.values()) or 1.0
    for v in stats.values():
        v["participation_share_pct"] = round((v["speaking_seconds"] / total) * 100, 1)
        minutes = v["speaking_seconds"] / 60
        v["wpm"] = round(v["words"] / minutes, 1) if minutes else 0
    return stats


def generate_gd_video_assessment(topic, participants, transcript_text, stats):
    prompt = f"""
You are a rigorous MBA Group Discussion evaluator for IPER Bhopal.

GD TOPIC:
{topic}

PARTICIPANTS:
{json.dumps(participants, ensure_ascii=False)}

OBJECTIVE SPEAKER STATISTICS:
{json.dumps(stats, ensure_ascii=False)}

DIARIZED TRANSCRIPT:
{transcript_text}

Evaluate ONLY what is supported by the transcript and objective statistics. Never invent facts, gestures, facial expressions, confidence, tone, eye contact, or knowledge that is not evidenced. Background music/noise is not speech. If a speaker has little or no transcript evidence, say so and do not fabricate a score.

For each participant evaluate:
- Participation: speaking share, number of meaningful interventions, balance, whether contributions add value.
- Communication: English fluency, grammar, vocabulary, clarity, filler words, rate of speech based on transcript/timestamps.
- Knowledge: relevance, factual/business awareness, examples, depth, understanding of the topic.
- Analytical Thinking: reasoning, cause-effect, comparison, trade-offs, originality.
- Listening & Team Behaviour: building on others, respectful disagreement, avoiding repetition/dominance, inviting others when evidenced.
- Leadership: initiative, structuring the discussion, synthesising ideas, moving toward conclusion when evidenced.
- Overall GD readiness.

Also evaluate the group as a whole:
- Topic handling
- Quality of discussion
- Balance of participation
- Depth of arguments
- Team dynamics
- Conclusion
- Overall group score

Return ONLY valid JSON in this exact structure:
{{
  "GroupAssessment": {{
    "OverallScore": 0,
    "TopicHandling": 0,
    "DiscussionQuality": 0,
    "ParticipationBalance": 0,
    "TeamDynamics": 0,
    "ConclusionQuality": 0,
    "Strengths": [],
    "Improvements": []
  }},
  "Participants": [
    {{
      "Name": "",
      "ParticipationScore": 0,
      "CommunicationScore": 0,
      "KnowledgeScore": 0,
      "AnalyticalThinkingScore": 0,
      "ListeningTeamworkScore": 0,
      "LeadershipScore": 0,
      "OverallScore": 0,
      "SpeakingTimeSeconds": 0,
      "ParticipationSharePercent": 0,
      "Interventions": 0,
      "WPM": 0,
      "FillerWords": 0,
      "Strengths": [],
      "AreasToImprove": [],
      "Evidence": [],
      "PlacementReadiness": "",
      "NextGDActionPlan": []
    }}
  ]
}}

All scores are integers from 0 to 10 except OverallScore fields, which are 0 to 100.
"""
    raw = get_groq_response(prompt)
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean), None
    except Exception:
        return None, raw


def save_gd_video_assessment(scholar_id, topic, participants, segments, report, video_filename):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO gd_assessments
        (scholar_id, topic, participant_names, transcript_json, report_json, video_filename, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        scholar_id, topic,
        json.dumps(participants, ensure_ascii=False),
        json.dumps(segments, ensure_ascii=False),
        json.dumps(report, ensure_ascii=False),
        video_filename,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_gd_video_assessments(scholar_id):
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT * FROM gd_assessments
        WHERE scholar_id=?
        ORDER BY id DESC
    """, (scholar_id,)).fetchall()
    conn.close()
    result = []
    for row in rows:
        item = dict(row)
        for key in ("participant_names", "transcript_json", "report_json"):
            try:
                item[key] = json.loads(item.get(key) or "[]")
            except Exception:
                item[key] = [] if key != "report_json" else {}
        result.append(item)
    return result


def _base64url_json(value):
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def create_video_engine_token(student, room_code, expires_minutes=15):
    """Create a short-lived HS256 JWT without adding another Python dependency."""
    if not VIDEO_ENGINE_JWT_SECRET:
        return None

    now = int(time.time())
    payload = {
        "studentId": str(student.get("id") or student.get("student_id") or student.get("scholar_id")),
        "firstName": str(student.get("first_name", "")),
        "lastName": str(student.get("last_name", "")),
        "scholarId": str(student.get("scholar_id", "")),
        "email": str(student.get("email", "")).lower(),
        "roomId": str(room_code).upper(),
        "iat": now,
        "exp": now + (expires_minutes * 60),
        "iss": "iper-placement-portal"
    }
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = _base64url_json(header)
    encoded_payload = _base64url_json(payload)
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(
        VIDEO_ENGINE_JWT_SECRET.encode("utf-8"),
        signing_input,
        digestmod="sha256"
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def render_inbuilt_gd_room(room_code, student, is_host=False, started_at=None):
    """Render the IPER-owned WebRTC room hosted by the separate video-engine service."""
    if not VIDEO_ENGINE_URL:
        st.error("The IPER Video Engine URL is not configured yet.")
        st.info("Add VIDEO_ENGINE_URL to Streamlit Secrets after deploying the video_engine service.")
        return

    if not VIDEO_ENGINE_JWT_SECRET:
        st.error("The IPER Video Engine security key is not configured yet.")
        st.info("Add VIDEO_ENGINE_JWT_SECRET to Streamlit Secrets. It must exactly match JWT_SECRET on the video engine service.")
        return

    token = create_video_engine_token(student, room_code, expires_minutes=15)
    if not token:
        st.error("Could not create a secure video-room access token.")
        return

    query = urllib.parse.urlencode({
        "room": str(room_code).upper(),
        "token": token,
        "firstName": student.get("first_name", ""),
        "lastName": student.get("last_name", ""),
        "scholarId": student.get("scholar_id", ""),
        "email": student.get("email", "")
    })
    room_url = f"{VIDEO_ENGINE_URL}/gd_room.html?{query}"
    display_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()
    role_label = "Student Host" if is_host else "Student Participant"

    st.markdown(f"""
    <div style="font-family:Inter,Arial,sans-serif;border:1px solid #CBD5E1;border-radius:12px;overflow:hidden;background:#0F172A;">
      <div style="padding:13px 16px;color:white;background:#0F172A;display:flex;justify-content:space-between;align-items:center;gap:12px;">
        <div><b>IPER Virtual GD Room</b><br><span style="font-size:12px;opacity:.85">{role_label} • {display_name} • maximum 7 students • maximum 10 minutes</span></div>
        <div style="font-size:12px;font-weight:700;color:#93C5FD;">IPER WebRTC</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.link_button("↗ Open IPER Virtual GD Room", room_url, use_container_width=True)
    st.caption("The secure room opens in a new browser tab so the camera and microphone can be granted directly to the IPER video service. Your name and Scholar ID are taken from your logged-in profile.")

    # Also provide an embedded view for browsers that allow camera/microphone access
    # inside the Streamlit component frame.
    components.html(f"""
    <div style="width:100%;height:720px;border:1px solid #CBD5E1;border-radius:12px;overflow:hidden;background:#0B1020;">
      <iframe
        src="{room_url}"
        title="IPER Virtual GD Room"
        allow="camera; microphone; fullscreen; display-capture"
        style="width:100%;height:100%;border:0;"
        allowfullscreen>
      </iframe>
    </div>
    """, height=735, scrolling=False)

def get_gd_ai_guidance(topic, student_name):
    prompt = f"""
You are an MBA Group Discussion coach at IPER Bhopal. Student name: {student_name}.
The student selected this GD topic: {topic}
Provide guidance, NOT a memorised speech. Cover exactly:
1. Topic interpretation
2. Arguments supporting the proposition
3. Arguments opposing the proposition
4. Business/economic perspective
5. Social perspective
6. Technology/future perspective
7. Indian context
8. Stakeholder perspectives
9. Three smart points a student could raise
10. Three questions a student could ask another participant
11. A balanced way to conclude
Use concise bullets and encourage listening and building on others' points.
"""
    return get_groq_response(prompt)


def generate_gd_feedback_ai(student_name, topic, voice, perspective, participation, camera, strengths, improvements, evidence=""):
    prompt = f"""
You are an MBA placement mentor at IPER Bhopal. Provide concise, constructive GD feedback for {student_name}.
Topic: {topic}
Self-ratings (1-10): Voice/Verbal Delivery={voice}, Perspective/Ideas={perspective}, Participation/Team Contribution={participation}, Camera Clarity/Visual Presence={camera}
Student strength reflection: {strengths}
Student improvement reflection: {improvements}
Concrete evidence supplied by the student: {evidence}
1. Voice — qualitative observation and one practical improvement.
2. Perspective — qualitative observation and one improvement.
3. Participation — qualitative observation and one improvement.
4. Camera Clarity — qualitative guidance on visual presence and technical clarity.
5. What You Did Well.
6. What You Should Improve.
7. Next GD Action Plan — exactly three actions.
Do not invent facts. When evidence is limited, clearly say the feedback is based on self-reflection.
"""
    return get_groq_response(prompt)

def generate_career_objective_and_power_words(resume_details, jd_text, student_name):
    prompt = f"""You are an MBA placement advisor at IPER Bhopal. Help {student_name} create a truthful, tailored career objective.
Use only information supported by the resume and job description. Do not invent experience.
Resume: {json.dumps(resume_details)}
Job Description: {jd_text}
Return ONLY valid JSON with keys: CareerObjectives (3 options), PowerWords (5 tailored words/phrases), WhereToUseThem (5 matching usage suggestions). Use simple, professional language for an MBA fresher."""
    return get_groq_response(prompt)

def generate_certification_advice(resume_details, jd_text, student_name):
    prompt = f"""You are an MBA placement advisor at IPER Bhopal. Assess whether certifications are genuinely useful for {student_name}'s target role. Do not recommend certificates just for collecting credentials.
Resume: {json.dumps(resume_details)}
Job Description: {jd_text}
Return ONLY valid JSON with keys: Necessity (Essential | Recommended | Optional | Not Required), Reason, RecommendedCourses (up to 3), SkillGap (up to 3), PriorityOrder. Use simple language and focus on practical placement value."""
    return get_groq_response(prompt)

def generate_learning_story(activity, activity_type, student_name, role, challenge, learning):
    prompt = f"""You are an MBA placement mentor at IPER Bhopal. Help {student_name} turn an activity into an interview-ready learning story. Activity: {activity}. Type: {activity_type}. Role: {role}. Challenge: {challenge}. Learning: {learning}. Use simple language and do not invent achievements. Return: 1. What interviewer wants to know 2. Simple STAR-style answer 3. Key learning 4. How this helps at work 5. One follow-up question."""
    return get_groq_response(prompt)

def render_authentication_panel():
    st.markdown(
        """
        <div style='text-align:center; padding: 30px 10px 10px 10px;'>
            <h1 style='margin-bottom:5px;'>IPER Student Placement Portal</h1>
            <p style='font-size:16px; color:#475569;'>Student Login & Career Readiness Hub</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab, reset_tab = st.tabs(["🔐 Student Sign In", "📝 Student Sign Up", "🔑 Forgot Password"])

    with login_tab:
        st.subheader("Sign in to your student account")
        with st.form("student_login_form"):
            login_email = st.text_input("Email ID", placeholder="yourname@iper.ac.in")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Sign In", use_container_width=True)

        if login_submitted:
            normalized_email = login_email.strip().lower()
            if not is_valid_iper_email(normalized_email):
                st.error("Please use your official @iper.ac.in email ID.")
            elif not login_password:
                st.error("Please enter your password.")
            else:
                student = authenticate_student(normalized_email, login_password)
                if student:
                    st.session_state["authenticated"] = True
                    st.session_state["student_id"] = student["id"]
                    st.session_state["student_email"] = student["email"]
                    st.session_state["first_name"] = student["first_name"]
                    st.session_state["last_name"] = student["last_name"]
                    st.session_state["scholar_id"] = student["scholar_id"]
                    st.session_state["candidate_name"] = student["first_name"]
                    st.session_state["history"] = load_student_attempts(student["id"])
                    st.session_state["resume_details"] = None
                    st.success(f"Welcome back, {student['first_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid email ID or password.")

    with reset_tab:
        st.subheader("Reset your password")
        st.info("Use the official @iper.ac.in email ID and Scholar ID registered with your account.")

        with st.form("student_password_reset_form"):
            reset_email = st.text_input("Registered Email ID", placeholder="yourname@iper.ac.in")
            reset_scholar_id = st.text_input("Scholar ID")
            reset_new_password = st.text_input("New Password", type="password")
            reset_confirm_password = st.text_input("Confirm New Password", type="password")
            reset_submitted = st.form_submit_button("Reset Password", use_container_width=True)

        if reset_submitted:
            normalized_reset_email = reset_email.strip().lower()

            if not is_valid_iper_email(normalized_reset_email):
                st.error("Please use your official @iper.ac.in email ID.")
            elif not reset_scholar_id.strip():
                st.error("Please enter your Scholar ID.")
            elif len(reset_new_password) < 8:
                st.error("New password must contain at least 8 characters.")
            elif reset_new_password != reset_confirm_password:
                st.error("New Password and Confirm New Password do not match.")
            else:
                try:
                    ok, message = reset_student_password(
                        normalized_reset_email, reset_scholar_id, reset_new_password
                    )
                    if ok:
                        st.success(message)
                        st.info("Please return to the Student Sign In tab and use your new password.")
                    else:
                        st.error(message)
                except Exception as exc:
                    st.error(f"Password reset could not be completed: {exc}")

    with signup_tab:
        st.subheader("Create your IPER student account")
        st.info("Only official @iper.ac.in email IDs can be registered.")

        with st.form("student_signup_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name")
                scholar_id = st.text_input("Scholar ID")
                email = st.text_input("Email ID (will be used as Login ID)", placeholder="yourname@iper.ac.in")
            with col2:
                last_name = st.text_input("Last Name")
                create_password = st.text_input("Create Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")

            signup_submitted = st.form_submit_button("Create Student Account", use_container_width=True)

        if signup_submitted:
            normalized_email = email.strip().lower()

            if not first_name.strip() or not last_name.strip():
                st.error("Please enter both First Name and Last Name.")
            elif not scholar_id.strip():
                st.error("Please enter your Scholar ID.")
            elif not is_valid_iper_email(normalized_email):
                st.error("Registration is restricted to official @iper.ac.in email IDs. Personal email IDs are not allowed.")
            elif len(create_password) < 8:
                st.error("Password must contain at least 8 characters.")
            elif create_password != confirm_password:
                st.error("Create Password and Confirm Password do not match.")
            else:
                ok, message = create_student(
                    first_name, last_name, scholar_id, normalized_email, create_password
                )
                if ok:
                    st.success(message)
                    st.info("Please open the Student Sign In tab and log in with your @iper.ac.in email ID.")
                else:
                    st.error(message)


init_database()
migrate_local_sqlite_to_postgres()

FFMPEG_PATH = shutil.which("ffmpeg")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    render_authentication_panel()
    st.stop()

# Keep the logged-in student's first name as the permanent personalized display name.
st.session_state["candidate_name"] = st.session_state.get("first_name", "Student")

# Personalized header shown throughout the logged-in student panel.
st.markdown(
    f"<div style='padding:8px 0 2px 0; font-size:18px; font-weight:600; color:#0F172A;'>Welcome, {st.session_state.get("first_name", "Student")} 👋</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 6. SIDEBAR NAVIGATION CONTROLS
# ------------------------------------------------------------------------------

st.sidebar.markdown("## IPER Student Portal")
st.sidebar.markdown(f"### Welcome, {st.session_state.get('first_name', 'Student')} 👋")
st.sidebar.caption(f"Scholar ID: {st.session_state.get('scholar_id', 'N/A')}")
st.sidebar.markdown("Career Readiness & Interview Hub")
if not FFMPEG_PATH:
    st.sidebar.warning("FFmpeg is not installed. Audio/video transcription will not work until FFmpeg is added to the deployment environment.")
st.sidebar.markdown("---")

selected_nav = st.sidebar.radio(
    "MAIN MENU",
    [
        "Resume Checker & Job Matcher", 
        "Career Development",
        "Interview Preparation Guide", 
        "Interview Practice Room", 
        "Group Discussion Hub",
        "Performance Dashboard"
    ]
)

if st.sidebar.button("Log Out", use_container_width=True):
    for key in ["authenticated", "student_id", "student_email", "first_name", "last_name", "scholar_id", "candidate_name", "resume_details", "current_question"]:
        st.session_state.pop(key, None)
    st.session_state["history"] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Recent Attempts**")
if not st.session_state["history"]:
    st.sidebar.info("No practice attempts recorded yet.")
else:
    for idx, session in enumerate(reversed(st.session_state["history"]), 1):
        with st.sidebar.expander(f"Session #{len(st.session_state['history']) - idx + 1}: {session['Timestamp']}"):
            st.markdown(f"**Candidate:** {session.get('Candidate', 'N/A')}")
            st.markdown(f"**Topic:** {session['Domain']}")
            st.markdown(f"**Score:** {session['Score']}/100")

# ------------------------------------------------------------------------------
# 6. APPLICATION SECTIONS
# ------------------------------------------------------------------------------

# SECTION 1: RESUME CHECKER & JOB MATCHER
if selected_nav == "Resume Checker & Job Matcher":
    st.title("Resume Checker & Job Matcher")
    st.caption("Upload your resume and a target job description to get clarity on your match level, key strengths, and areas to polish.")

    col_res, col_jd = st.columns(2)

    with col_res:
        st.subheader("1. Candidate Resume Upload")
        ats_resume_file = st.file_uploader("Upload Resume (PDF or Word DOCX):", type=["pdf", "docx", "doc"], key="ats_res_upload")
        
        if ats_resume_file:
            res_raw_text = extract_text_from_file(ats_resume_file)
            if res_raw_text and not res_raw_text.startswith("Error"):
                parse_prompt = f"""
                Analyze the following resume text and return STRICT JSON with exact keys:
                {{
                    "Name": "<Candidate Full Name>",
                    "Education": "<Degrees, Institutions, Specializations>",
                    "Work_Experience": "<Total years, key roles, companies>",
                    "Key_Skills": "<Technical and functional skills>",
                    "Projects_Achievements": "<Key projects, research, or certifications>",
                    "Summary": "<2-3 sentence overview>"
                }}
                Do not include commentary or markdown wrapping outside valid JSON.
                Resume Text:
                {res_raw_text[:3500]}
                """
                parsed_raw = get_groq_response(parse_prompt)
                try:
                    clean_json = parsed_raw.replace("```json", "").replace("```", "").strip()
                    resume_data = json.loads(clean_json)
                    c_name = resume_data.get("Name", "Candidate")
                except Exception:
                    c_name = "Candidate"
                    resume_data = {"Summary": parsed_raw, "RawText": res_raw_text[:2000]}
                
                st.session_state["resume_details"] = resume_data
                st.success(f"Resume loaded successfully for **{st.session_state.get('first_name', 'Student')}**")

    with col_jd:
        st.subheader("2. Target Job Description")
        jd_input_option = st.radio("Provide Job Description via:", ["Upload File (PDF/DOCX/Image)", "Paste Text Direct"], horizontal=True)
        
        jd_text = ""
        if jd_input_option == "Upload File (PDF/DOCX/Image)":
            jd_file = st.file_uploader("Upload Job Description File:", type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], key="ats_jd_upload")
            if jd_file:
                jd_text = extract_text_from_file(jd_file)
                st.info(f"Loaded File: `{jd_file.name}`")
        else:
            jd_text = st.text_area("Paste Job Description Text Here:", height=180, placeholder="Paste requirements, job duties, and key qualifications...")

    st.markdown("---")

    if st.button("Check Match & Review Suggestions", use_container_width=True):
        if not ats_resume_file:
            st.error("Please upload your candidate resume first.")
        elif not jd_text.strip():
            st.error("Please upload or paste a Job Description.")
        else:
            with st.spinner("Reviewing match and alignment..."):
                candidate_name = st.session_state.get("first_name", "Student")
                resume_content = json.dumps(st.session_state.get("resume_details", {}))
                
                ats_prompt = f"""
                Act as a helpful Corporate Recruiter & Placement Reviewer at IPER Bhopal.
                Evaluate the resume alignment for candidate '{candidate_name}'.

                Resume Information:
                {resume_content}

                Job Description Information:
                {jd_text}

                Return ONLY valid JSON with this exact structure:
                {{
                    "CandidateName": "{candidate_name}",
                    "ATSScore": <0-100 integer score>,
                    "Category": "<MUST be exactly one of: High Match | Good Match | Needs Improvement>",
                    "ExecutiveSummary": "<Greeting addressing {candidate_name} by name with a friendly summary of overall fit>",
                    "Strengths": ["<Strength 1>", "<Strength 2>", "<Strength 3>"],
                    "AreasOfImprovement": ["<Area 1>", "<Area 2>", "<Area 3>"],
                    "RecommendedChanges": ["<Specific bullet update 1>", "<Keyword addition 2>", "<Formatting tip 3>"],
                    "MissingKeywords": ["<Keyword 1>", "<Keyword 2>", "<Keyword 3>"]
                }}
                """
                raw_ats_response = get_groq_response(ats_prompt)
                
                try:
                    clean_ats = raw_ats_response.replace("```json", "").replace("```", "").strip()
                    ats_result = json.loads(clean_ats)
                    
                    st.subheader(f"Resume Analysis for {candidate_name}")
                    
                    score = ats_result.get("ATSScore", 70)
                    category_rating = ats_result.get("Category", "Good Match")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Overall Match Score", f"{score} / 100")
                    with m2:
                        st.metric("Fit Status", category_rating)
                    
                    st.write(ats_result.get("ExecutiveSummary", ""))
                    st.markdown("---")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Profile Strengths")
                        for str_item in ats_result.get("Strengths", []):
                            st.markdown(f"- **{str_item}**")
                            
                        st.markdown("### Suggested Keywords to Add")
                        for kw in ats_result.get("MissingKeywords", []):
                            st.markdown(f"- `{kw}`")
                            
                    with c2:
                        st.markdown("### Areas to Improve")
                        for imp in ats_result.get("AreasOfImprovement", []):
                            st.markdown(f"- {imp}")
                            
                        st.markdown("### Practical Recommendations")
                        for chg in ats_result.get("RecommendedChanges", []):
                            st.markdown(f"- {chg}")

                except Exception:
                    st.markdown(raw_ats_response)

# SECTION 2: CAREER DEVELOPMENT
elif selected_nav == "Career Development":
    st.title("Career Development")
    st.caption("Build a tailored career objective, find powerful resume words, and understand whether certifications are actually needed.")
    if not st.session_state.get("resume_details"):
        st.info("Upload your resume in Resume Checker & Job Matcher first.")
    career_jd = st.text_area("Target Job Description", height=180, placeholder="Paste the target JD here...")
    tab1, tab2 = st.tabs(["🎯 Career Objective & Power Words", "📜 Certification Necessity"])
    with tab1:
        if st.button("Generate Tailored Career Objective", use_container_width=True):
            if not st.session_state.get("resume_details") or not career_jd.strip():
                st.warning("Please upload your resume and paste the target Job Description.")
            else:
                with st.spinner("Creating tailored objectives and power words..."):
                    raw = generate_career_objective_and_power_words(st.session_state["resume_details"], career_jd, st.session_state.get("first_name", "Student"))
                try:
                    data=json.loads(raw.replace("```json","").replace("```","").strip())
                    st.markdown("### Career Objective Options")
                    for i,x in enumerate(data.get("CareerObjectives",[]),1): st.info(f"**Option {i}:** {x}")
                    st.markdown("### Powerful Words Tailored to the JD")
                    for word,use in zip(data.get("PowerWords",[]),data.get("WhereToUseThem",[])): st.markdown(f"- **{word}** — {use}")
                except Exception: st.markdown(raw)
    with tab2:
        if st.button("Assess Certification Necessity", use_container_width=True):
            if not st.session_state.get("resume_details") or not career_jd.strip():
                st.warning("Please upload your resume and paste the target Job Description.")
            else:
                with st.spinner("Assessing skill gaps and certification value..."):
                    raw=generate_certification_advice(st.session_state["resume_details"],career_jd,st.session_state.get("first_name","Student"))
                try:
                    data=json.loads(raw.replace("```json","").replace("```","").strip())
                    st.metric("Certification Necessity",data.get("Necessity","Not Required")); st.write(data.get("Reason",""))
                    st.markdown("### Recommended Courses / Certifications")
                    for x in data.get("RecommendedCourses",[]): st.markdown(f"- {x}")
                    st.markdown("### Skill Gaps")
                    for x in data.get("SkillGap",[]): st.markdown(f"- {x}")
                    st.markdown("### Priority Order")
                    for i,x in enumerate(data.get("PriorityOrder",[]),1): st.markdown(f"{i}. {x}")
                except Exception: st.markdown(raw)

# SECTION 2: INTERVIEW PREPARATION GUIDE
elif selected_nav == "Interview Preparation Guide":
    st.title("Interview Preparation Guide")
    st.caption("Prepare with simple, placement-focused questions, clear answer structures, and practical examples.")
    tabs=st.tabs(["⭐ Core Placement Questions","📚 Subject & Company Preparation","🎓 What Did I Learn?"])
    with tabs[0]:
        st.markdown("### Essential Interview Questions")
        st.info("Practise these questions in your own words. Do not memorise the model answer.")
        core_q=st.selectbox("Select a core question",CORE_INTERVIEW_QUESTIONS,key="core_interview_q")
        if st.button("Prepare This Question",key="prepare_core_q",use_container_width=True):
            with st.spinner("Preparing a simple interview guide..."):
                prompt=f"""Act as a supportive MBA placement mentor at IPER Bhopal. Question: {core_q}. Use very simple, natural Indian-English. Avoid jargon and textbook language. Explain: 1) what the interviewer wants to know, 2) a simple answer structure, 3) a short natural sample answer, 4) one mistake to avoid, 5) one follow-up question."""
                st.markdown(get_groq_response(prompt))
    with tabs[1]:
        prep_category=st.selectbox("Select Study Domain:",["General and Core Skills"]+SPECIALIZATIONS+["Company Specific"],key="prep_category_main")
        if prep_category=="Company Specific":
            comp_choice=st.selectbox("Select Target Company:",IPER_RECRUITERS,key="prep_company_main")
            if st.button("Load Top Recruiter Questions",key="load_recruiter_questions"):
                with st.spinner(f"Retrieving placement-style questions for {comp_choice}..."):
                    st.markdown(get_groq_response(f"Generate 5 simple, realistic technical and situational interview questions for {comp_choice} during campus hiring. Use clear MBA-student language."))
        else:
            q_list=EXHAUSTIVE_QUESTIONS.get(prep_category,["Describe a key challenge you faced and how you resolved it."])
            selected_question=st.selectbox("Select Question to Study",q_list,key="prep_question_main")
            st.markdown(f"### Study Guide: {selected_question}")
            if st.button("Generate Simple Answer Framework & Model Answer",key="generate_simple_guide"):
                with st.spinner("Preparing a simple answer breakdown..."):
                    prompt=f"""Act as a senior MBA Placement Advisor at IPER Bhopal. Question: {selected_question}. Domain: {prep_category}. Use simple natural language and avoid unnecessary jargon. Provide: 1) what interviewers look for, 2) simple STAR/CAR structure where appropriate, 3) short natural model answer, 4) one mistake to avoid, 5) one follow-up question."""
                    st.markdown(get_groq_response(prompt))
    with tabs[2]:
        st.markdown("### Turn Activities into Interview Stories")
        activity_type=st.radio("Activity Type",["Curricular","Extra-Curricular"],horizontal=True)
        activity_list=CURRICULAR_ACTIVITIES if activity_type=="Curricular" else EXTRA_CURRICULAR_ACTIVITIES
        activity=st.selectbox("Select Activity",activity_list,key="learning_activity")
        role=st.text_input("Your role",placeholder="Example: Team leader / participant / coordinator")
        challenge=st.text_area("What challenge did you face?",height=90,key="learning_challenge")
        learning=st.text_area("What did you learn?",height=90,key="learning_text")
        if st.button("Create Interview-Ready Learning Story",key="create_learning_story",use_container_width=True):
            if not role.strip() or not learning.strip(): st.warning("Please add your role and what you learned.")
            else:
                with st.spinner("Converting your experience into a simple interview answer..."):
                    st.markdown(generate_learning_story(activity,activity_type,st.session_state.get("first_name","Student"),role,challenge,learning))

# SECTION 3: INTERVIEW PRACTICE ROOM
elif selected_nav == "Interview Practice Room":
    st.title("Interview Practice Room")
    st.caption("Practice audio or video interview responses and receive structured, constructive feedback.")
    
    if st.session_state.get("resume_details"):
        st.markdown(f"**Active Student:** {st.session_state.get('first_name', 'Student')} *(Resume loaded)*")
    else:
        st.markdown("*(Tip: You can upload your resume in the **Resume Checker** section for custom questions.)*")

    st.markdown("---")
    col_mode, col_diff, col_cat = st.columns(3)
    with col_mode:
        mode = st.radio("Practice Format:", ["Audio Response Mode", "Video Response Mode"])
    with col_diff:
        difficulty = st.selectbox("Question Level:", ["Basic", "Intermediate", "Advanced"])
    with col_cat:
        category = st.selectbox("Question Focus:", ["Resume-Based (Custom)", "General and Core Skills", "Specialization", "Company Specific"])

    selected_spec = None
    selected_comp = None
    if category == "Specialization":
        selected_spec = st.selectbox("Select Specialization Track:", SPECIALIZATIONS)
    elif category == "Company Specific":
        selected_comp = st.selectbox("Select Target Recruiter:", IPER_RECRUITERS)

    if st.button("Get Practice Question", use_container_width=True):
        if category == "Resume-Based (Custom)":
            if not st.session_state.get("resume_details"):
                st.warning("Please upload your resume in the Resume Checker section first to generate custom questions.")
            else:
                prompt = f"""
                Act as a supportive MBA interviewer at IPER Bhopal. 
                Generate ONE interview question tailored to {st.session_state.get('first_name', 'Student')}'s resume profile:
                {json.dumps(st.session_state['resume_details'])}
                Difficulty level: {difficulty}
                Return ONLY the question text clearly.
                """
                with st.spinner("Generating question..."):
                    st.session_state["current_question"] = get_groq_response(prompt)
        else:
            ctx = f"Domain: {category}, Level: {difficulty}"
            if selected_spec: ctx += f", Track: {selected_spec}"
            if selected_comp: ctx += f", Target Company: {selected_comp}"
            if st.session_state["resume_details"]: ctx += f", Candidate Context: {json.dumps(st.session_state['resume_details'])}"
            
            prompt = f"Act as an MBA interviewer at IPER Bhopal. Generate ONE interview question for '{st.session_state.get('first_name', 'Student')}' based on: {ctx}. Return ONLY question text."
            with st.spinner("Retrieving question..."):
                st.session_state["current_question"] = get_groq_response(prompt)

    if "current_question" in st.session_state:
        st.markdown(f"### Question for {st.session_state.get('first_name', 'Student')}")
        st.write(f"*{st.session_state['current_question']}*")
        st.markdown("---")

        extracted_transcript = ""
        saved_video_filename = "N/A"
        st.session_state["speech_detected"] = False

        if mode == "Audio Response Mode":
            st.subheader("Record Your Audio Answer")
            audio_data = st.audio_input("Click to record your voice:")
            if audio_data is not None:
                st.audio(audio_data)
                with st.spinner("Converting audio to text via Whisper..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_data.getvalue())
                        tmp_path = tmp_file.name
                    try:
                        st.session_state["communication_duration"] = get_media_duration_seconds(tmp_path)
                        extracted_transcript, detected_speech_duration = transcribe_indian_english_audio(tmp_path)
                        if detected_speech_duration > 0:
                            st.session_state["communication_duration"] = detected_speech_duration
                    except Exception as err:
                        st.error(f"Speech transcription error: {err}")
                        if not FFMPEG_PATH:
                            st.info("FFmpeg is missing. Add `ffmpeg` to packages.txt (Streamlit Cloud) or install it in your Dockerfile (Cloud Run).")
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                
                if extracted_transcript:
                    st.session_state["speech_detected"] = bool(extracted_transcript.strip())
                    if extracted_transcript:
                        st.success("Human speech detected and transcribed successfully.")
                    else:
                        st.warning("No clear human speech was detected. Background music/noise was ignored, so no AI communication feedback was generated.")

        elif mode == "Video Response Mode":
            st.subheader("1. Record & Save Video")
            render_video_recorder_component()
            
            st.subheader("2. Upload Saved Video File")
            uploaded_video = st.file_uploader("Upload recorded video file (.webm or .mp4):", type=["webm", "mp4"])
            
            if uploaded_video is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                incoming_ext = "mp4" if uploaded_video.name.lower().endswith(".mp4") else "webm"
                incoming_path = os.path.join(VIDEO_STORAGE_DIR, f"incoming_{ts}.{incoming_ext}")
                saved_video_path = os.path.join(VIDEO_STORAGE_DIR, f"video_{ts}.mp4")
                
                with open(incoming_path, "wb") as f:
                    f.write(uploaded_video.getvalue())
                
                try:
                    with st.spinner("Preparing MP4 interview video..."):
                        if incoming_ext == "webm":
                            saved_video_path = convert_video_to_mp4(incoming_path)
                            if os.path.abspath(saved_video_path) != os.path.abspath(incoming_path):
                                # convert_video_to_mp4 uses the input stem, so rename it to the final naming convention.
                                target = os.path.join(VIDEO_STORAGE_DIR, f"video_{ts}.mp4")
                                if os.path.abspath(saved_video_path) != os.path.abspath(target):
                                    os.replace(saved_video_path, target)
                                saved_video_path = target
                        else:
                            with open(saved_video_path, "wb") as out:
                                out.write(uploaded_video.getvalue())
                    if os.path.exists(incoming_path) and os.path.abspath(incoming_path) != os.path.abspath(saved_video_path):
                        os.remove(incoming_path)
                    st.session_state["current_video_path"] = saved_video_path
                    st.session_state["communication_duration"] = get_media_duration_seconds(saved_video_path)
                    st.video(saved_video_path)
                    st.success(f"Interview video saved as MP4: `{os.path.basename(saved_video_path)}`")
                    with st.spinner("Transcribing video audio track via Whisper..."):
                        extracted_transcript, detected_speech_duration = transcribe_indian_english_audio(saved_video_path)
                        if detected_speech_duration > 0:
                            st.session_state["communication_duration"] = detected_speech_duration
                        st.session_state["speech_detected"] = bool(extracted_transcript.strip())
                        if extracted_transcript:
                            st.success("Human speech detected and transcribed successfully.")
                        else:
                            st.warning("No clear human speech was detected. Background music/noise was ignored, so no AI communication feedback was generated.")
                except Exception as err:
                    st.error(f"Video processing error: {err}")
                    if os.path.exists(incoming_path):
                        try: os.remove(incoming_path)
                        except OSError: pass

        final_response_text = st.text_area(
            "Answer Transcript (Review or Edit Text Before Submitting):", 
            value=extracted_transcript, 
            height=140
        )

        if st.button("Submit Response for Feedback"):
            if not final_response_text.strip():
                st.error("Please record your audio/video response or write your transcript text first.")
            elif mode in ("Audio Response Mode", "Video Response Mode") and not st.session_state.get("speech_detected", False):
                st.error("No clear human speech was detected in this recording. Background music/noise has been ignored. Please record a spoken answer, or enter the transcript manually if you intentionally want to evaluate typed text.")
            else:
                with st.spinner("Analyzing your response..."):
                    c_name = st.session_state.get('first_name', 'Student')
                    st.session_state["communication_metrics"] = communication_metrics(
                        final_response_text, st.session_state.get("communication_duration", 0.0)
                    )
                    eval_prompt = f"""
                    Act as an encouraging MBA placement mentor at IPER Bhopal.
                    Evaluate the response for candidate: {c_name}.
                    Address {c_name} warmly by name across feedback areas.

                    Question Asked: {st.session_state['current_question']}
                    Candidate Response: {final_response_text}
                    Resume Context: {json.dumps(st.session_state.get('resume_details', {}))}
                    Practice Mode: {mode}

                    Communication metrics calculated from the recording/transcript:
                    {json.dumps(st.session_state.get("communication_metrics", {}))}

                    Evaluate the candidate as an MBA placement interviewer and English communication coach.
                    Assess English communication separately from subject knowledge. Be constructive, evidence-based, and do not invent observations that cannot be supported by the transcript or calculated metrics.

                    Return response in VALID JSON strictly matching this structure:
                    {{
                        "CandidateName": "{c_name}",
                        "GradingScore": <0-100 integer>,
                        "ExecutiveSummary": "<Warm overall summary>",
                        "TechnicalAssessment": "<Evaluation of subject knowledge and key points>",
                        "CommunicationAssessment": "<Overall English communication assessment>",
                        "EnglishCommunicationScore": <0-10>,
                        "GrammarScore": <0-10>,
                        "FillerWordsScore": <0-10>,
                        "RateOfSpeechScore": <0-10>,
                        "ToneScore": <0-10>,
                        "ClarityScore": <0-10>,
                        "GrammarIssues": ["<specific grammar issue and corrected form>"],
                        "FillerWordsUsed": {{"word_or_phrase": <count>}},
                        "RateOfSpeechAssessment": "<Assess WPM against professional interview speaking: generally around 120-160 WPM is a useful reference, but context matters>",
                        "ToneAssessment": "<Assess professionalism, confidence, warmth, hesitation, and appropriateness from available evidence. Do not claim to hear vocal tone if only transcript is available>",
                        "ClarityAssessment": "<Assess structure, sentence clarity, coherence, and ease of understanding>",
                        "CommunicationStrengths": ["<strength>"],
                        "CommunicationImprovements": ["<improvement>"],
                        "ResumeAlignment": "<How effectively the candidate referenced relevant experience>",
                        "KeyFlaws": "<Constructive highlights of points missed or needing clarity>",
                        "CorrectiveSteps": "<Practical tips for {c_name} to improve next time>",
                        "Benchmark100Answer": "<A clear, exemplary benchmark model answer>"
                    }}
                    """
                    raw_eval = get_groq_response(eval_prompt)
                    
                    try:
                        clean_json = raw_eval.replace("```json", "").replace("```", "").strip()
                        eval_result = json.loads(clean_json)

                        score = int(eval_result.get("GradingScore", 0))
                        score = max(0, min(100, score))

                        st.subheader(f"Interview Feedback for {c_name}")
                        st.metric("Overall Score", f"{score} / 100")

                        metrics = st.session_state.get("communication_metrics", {})
                        if metrics:
                            st.markdown("### 🎙️ English Communication Analysis")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("English Communication", f"{eval_result.get('EnglishCommunicationScore', 0)}/10")
                            m2.metric("Grammar", f"{eval_result.get('GrammarScore', 0)}/10")
                            m3.metric("Rate of Speech", f"{metrics.get('wpm', 0)} WPM")
                            m4.metric("Filler Words", str(metrics.get('filler_total', 0)))

                            m5, m6, m7 = st.columns(3)
                            m5.metric("Tone", f"{eval_result.get('ToneScore', 0)}/10")
                            m6.metric("Clarity", f"{eval_result.get('ClarityScore', 0)}/10")
                            m7.metric("Duration", f"{metrics.get('duration_minutes', 0)} min")

                            st.markdown("**Grammar feedback**")
                            for issue in eval_result.get("GrammarIssues", []):
                                st.markdown(f"- {issue}")
                            st.markdown("**Filler words detected**")
                            filler = metrics.get("filler_counts", {})
                            st.write(filler if filler else "No common filler words detected.")
                            st.markdown("**Rate of speech**")
                            st.write(eval_result.get("RateOfSpeechAssessment", ""))
                            st.markdown("**Tone**")
                            st.write(eval_result.get("ToneAssessment", ""))
                            st.markdown("**Clarity**")
                            st.write(eval_result.get("ClarityAssessment", ""))
                            st.markdown("**Communication strengths**")
                            for item in eval_result.get("CommunicationStrengths", []):
                                st.markdown(f"- {item}")
                            st.markdown("**Communication improvements**")
                            for item in eval_result.get("CommunicationImprovements", []):
                                st.markdown(f"- {item}")

                        st.write(eval_result.get("ExecutiveSummary", ""))

                        st.markdown("### Technical Assessment")
                        st.write(eval_result.get("TechnicalAssessment", ""))

                        st.markdown("### Communication Assessment")
                        st.write(eval_result.get("CommunicationAssessment", ""))

                        st.markdown("### Resume Alignment")
                        st.write(eval_result.get("ResumeAlignment", ""))

                        st.markdown("### Key Areas to Improve")
                        st.write(eval_result.get("KeyFlaws", ""))

                        st.markdown("### Corrective Steps")
                        st.write(eval_result.get("CorrectiveSteps", ""))

                        st.markdown("### Benchmark 100/100 Answer")
                        st.write(eval_result.get("Benchmark100Answer", ""))

                        attempt_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        attempt = {
                            "Timestamp": attempt_timestamp,
                            "Candidate": c_name,
                            "Domain": category,
                            "Score": score,
                            "Mode": mode,
                            "Question": st.session_state.get("current_question", "")
                        }
                        st.session_state["history"].append(attempt)
                        save_student_attempt(
                            st.session_state["student_id"],
                            category,
                            score,
                            mode,
                            st.session_state.get("current_question", "")
                        )

                    except (json.JSONDecodeError, TypeError, ValueError):
                        st.warning("The AI response was not returned as valid JSON. The raw feedback is shown below.")
                        st.markdown(raw_eval)

# SECTION 4: GROUP DISCUSSION HUB
elif selected_nav == "Group Discussion Hub":
    st.title("Group Discussion Hub")
    st.caption("Conduct the GD on Google Meet, Zoom, Teams or a phone. Upload the completed video here and let AI evaluate the discussion.")

    gd_prep_tab, gd_practice_tab, gd_feedback_tab = st.tabs([
        "📚 GD Preparation", "🎥 Upload GD Video", "📊 My GD Feedback"
    ])

    with gd_prep_tab:
        prep_col1, prep_col2 = st.columns(2)
        with prep_col1:
            st.markdown("### ✅ Do's")
            for item in GD_DOS:
                st.markdown(f"- {item}")
        with prep_col2:
            st.markdown("### ❌ Don'ts")
            for item in GD_DONTS:
                st.markdown(f"- {item}")

        st.markdown("---")
        st.markdown("### 100+ GD Topics")
        topic = st.selectbox("Select a GD topic for AI perspective guidance:", GD_TOPICS)
        if st.button("Get AI Perspective Guidance", use_container_width=True):
            with st.spinner("Preparing balanced perspectives..."):
                st.markdown(get_gd_ai_guidance(topic, st.session_state.get("first_name", "Student")))

    with gd_practice_tab:
        st.markdown("### 🎥 IPER AI GD Assessment")
        st.info("Conduct your GD on any third-party platform, download the recording as MP4, then upload it here. The portal analyses participation, communication, knowledge, analytical thinking, teamwork and leadership.")

        if not OPENAI_API_KEY:
            st.warning("For multi-speaker identification, add OPENAI_API_KEY to Streamlit Secrets. The GD transcription uses speaker diarization so individual participation can be measured accurately.")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY is required for the final GD evaluation.")

        with st.form("gd_video_upload_form"):
            upload_topic = st.selectbox("GD Topic", GD_TOPICS, key="gd_upload_topic")
            participant_text = st.text_area(
                "Participant names — one per line (maximum 7)",
                placeholder="Rahul Sharma\nPriya Jain\nAnanya Singh\nArjun Patel"
            )
            gd_video = st.file_uploader(
                "Upload completed GD video",
                type=["mp4", "webm", "mov", "m4v", "mpeg", "mpg"],
                help="MP4 is recommended. Maximum practical duration: 10 minutes."
            )
            consent = st.checkbox("I confirm that all participants have agreed to this recording being used for educational assessment.")
            process_gd = st.form_submit_button("🚀 Analyse GD", use_container_width=True)

        if process_gd:
            participants = [x.strip() for x in participant_text.splitlines() if x.strip()]
            participants = participants[:GD_MAX_PARTICIPANTS]

            if len(participants) < 2:
                st.error("Please enter at least 2 participant names.")
            elif not gd_video:
                st.error("Please upload the GD video.")
            elif not consent:
                st.error("Please confirm participant consent before processing.")
            elif not OPENAI_API_KEY:
                st.error("OPENAI_API_KEY is required for speaker-diarized GD analysis.")
            else:
                suffix = Path(gd_video.name).suffix.lower() or ".mp4"
                video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                video_tmp.write(gd_video.getbuffer())
                video_tmp.close()
                audio_path = None

                try:
                    with st.status("Processing GD recording...", expanded=True) as status:
                        st.write("1/4 Extracting clean mono audio with FFmpeg...")
                        audio_path = _ffmpeg_extract_gd_audio(video_tmp.name)
                        if not audio_path:
                            raise RuntimeError("FFmpeg could not extract audio from this video.")

                        st.write("2/4 Detecting speakers and transcribing human speech...")
                        segments, error = transcribe_gd_with_diarization(audio_path)
                        if error:
                            raise RuntimeError(error)

                        detected_speakers = sorted(
                            {s["speaker"] for s in segments},
                            key=lambda x: x
                        )
                        st.write(f"Detected {len(detected_speakers)} speaker(s).")

                        st.write("3/4 Preparing objective participation metrics...")
                        default_map = {}
                        if len(detected_speakers) == len(participants):
                            for s, name in zip(detected_speakers, participants):
                                default_map[s] = name

                        st.write("4/4 Generating comprehensive AI assessment...")
                        st.session_state["gd_pending_analysis"] = {
                            "topic": upload_topic,
                            "participants": participants,
                            "segments": segments,
                            "speaker_map": default_map,
                            "video_filename": gd_video.name
                        }
                        status.update(label="GD transcription complete — confirm speaker mapping below.", state="complete")

                except Exception as exc:
                    st.error(f"GD processing failed: {exc}")
                finally:
                    try:
                        os.remove(video_tmp.name)
                    except OSError:
                        pass
                    if audio_path:
                        try:
                            os.remove(audio_path)
                        except OSError:
                            pass

        pending = st.session_state.get("gd_pending_analysis")
        if pending:
            st.markdown("---")
            st.markdown("### 👥 Match Detected Speakers to Students")
            st.caption("This confirmation step makes the individual reports more reliable. If the recording contains fewer detected speakers than the participant list, leave unused students as Unassigned.")

            detected = sorted({s["speaker"] for s in pending["segments"]})
            options = ["Unassigned"] + pending["participants"]
            mapping = {}
            cols = st.columns(min(3, max(1, len(detected))))
            for i, speaker in enumerate(detected):
                default_name = pending["speaker_map"].get(speaker, "Unassigned")
                default_index = options.index(default_name) if default_name in options else 0
                with cols[i % len(cols)]:
                    selected_name = st.selectbox(
                        speaker,
                        options,
                        index=default_index,
                        key=f"gd_map_{speaker}"
                    )
                    mapping[speaker] = selected_name

            if len([v for v in mapping.values() if v != "Unassigned"]) != len(set(v for v in mapping.values() if v != "Unassigned")):
                st.warning("Two detected speakers are mapped to the same student. Please give each student a unique speaker.")

            stats = _gd_speaker_stats(
                pending["segments"],
                {k: v for k, v in mapping.items() if v != "Unassigned"}
            )
            st.markdown("#### Objective participation snapshot")
            snapshot_rows = []
            for speaker, vals in stats.items():
                snapshot_rows.append({
                    "Participant": speaker,
                    "Speaking Time": f"{vals['speaking_seconds']:.0f}s",
                    "Share": f"{vals['participation_share_pct']:.1f}%",
                    "Interventions": vals["turns"],
                    "WPM": vals["wpm"],
                    "Filler Words": vals["filler_words"]
                })
            if snapshot_rows:
                st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True, hide_index=True)

            if st.button("🧠 Generate Final GD Report", use_container_width=True):
                clean_map = {k: v for k, v in mapping.items() if v != "Unassigned"}
                transcript_text = _format_gd_transcript(pending["segments"], clean_map)
                with st.spinner("AI is evaluating participation, communication, knowledge, analytical thinking and teamwork..."):
                    report, raw_or_error = generate_gd_video_assessment(
                        pending["topic"],
                        pending["participants"],
                        transcript_text,
                        stats
                    )
                if report:
                    mapped_segments = [
                        {**seg, "speaker": clean_map.get(seg["speaker"], seg["speaker"])}
                        for seg in pending["segments"]
                    ]
                    save_gd_video_assessment(
                        st.session_state["scholar_id"],
                        pending["topic"],
                        pending["participants"],
                        mapped_segments,
                        report,
                        pending["video_filename"]
                    )
                    st.session_state["last_gd_report"] = report
                    st.session_state.pop("gd_pending_analysis", None)
                    st.success("Complete GD assessment generated and saved to your profile.")
                    st.rerun()
                else:
                    st.error("The AI report could not be parsed as JSON.")
                    st.code(raw_or_error or "", language="text")

    with gd_feedback_tab:
        st.markdown("### 📊 My GD Feedback")
        st.caption("Your uploaded GD recordings are converted into objective speaker statistics and a comprehensive placement-style assessment.")

        assessments = get_gd_video_assessments(st.session_state["scholar_id"])
        latest = assessments[0] if assessments else None

        if latest:
            report = latest.get("report_json") or {}
            group = report.get("GroupAssessment", {})
            st.markdown(f"### Latest GD — {latest.get('topic', '')}")
            st.caption(f"Processed on {latest.get('created_at', '')} • Source video: {latest.get('video_filename', '')}")

            gcols = st.columns(5)
            for col, label, key in zip(
                gcols,
                ["Overall", "Topic", "Discussion", "Balance", "Team Dynamics"],
                ["OverallScore", "TopicHandling", "DiscussionQuality", "ParticipationBalance", "TeamDynamics"]
            ):
                col.metric(label, f"{group.get(key, 0)}/100")

            if group.get("Strengths"):
                st.markdown("#### 🟢 Group Strengths")
                for item in group["Strengths"]:
                    st.write(f"• {item}")
            if group.get("Improvements"):
                st.markdown("#### 🟠 Group Improvements")
                for item in group["Improvements"]:
                    st.write(f"• {item}")

            st.markdown("### 👤 Individual Performance")
            for person in report.get("Participants", []):
                with st.expander(f"{person.get('Name', 'Participant')} — {person.get('OverallScore', 0)}/100", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Participation", f"{person.get('ParticipationScore', 0)}/10")
                    c2.metric("Communication", f"{person.get('CommunicationScore', 0)}/10")
                    c3.metric("Knowledge", f"{person.get('KnowledgeScore', 0)}/10")
                    c4.metric("Leadership", f"{person.get('LeadershipScore', 0)}/10")

                    c5, c6, c7 = st.columns(3)
                    c5.metric("Analytical Thinking", f"{person.get('AnalyticalThinkingScore', 0)}/10")
                    c6.metric("Listening & Teamwork", f"{person.get('ListeningTeamworkScore', 0)}/10")
                    c7.metric("WPM", person.get("WPM", 0))

                    st.write(
                        f"**Speaking time:** {person.get('SpeakingTimeSeconds', 0):.0f}s  • "
                        f"**Participation share:** {person.get('ParticipationSharePercent', 0)}%  • "
                        f"**Interventions:** {person.get('Interventions', 0)}  • "
                        f"**Filler words:** {person.get('FillerWords', 0)}"
                    )

                    if person.get("Strengths"):
                        st.markdown("**Strengths**")
                        for item in person["Strengths"]:
                            st.write(f"• {item}")
                    if person.get("AreasToImprove"):
                        st.markdown("**Areas to improve**")
                        for item in person["AreasToImprove"]:
                            st.write(f"• {item}")
                    if person.get("Evidence"):
                        st.markdown("**Evidence from the recording**")
                        for item in person["Evidence"]:
                            st.write(f"• {item}")
                    if person.get("PlacementReadiness"):
                        st.markdown("**Placement readiness**")
                        st.write(person["PlacementReadiness"])
                    if person.get("NextGDActionPlan"):
                        st.markdown("**Next GD action plan**")
                        for item in person["NextGDActionPlan"]:
                            st.write(f"• {item}")

            with st.expander("📝 Diarized Transcript", expanded=False):
                st.text(_format_gd_transcript(
                    latest.get("transcript_json") or [],
                    {}
                ))

            st.markdown("### Previous GD Assessments")
            for old in assessments[1:10]:
                st.write(f"• {old.get('created_at')} — {old.get('topic')} — {old.get('video_filename')}")
        else:
            st.info("No GD assessment yet. Upload a completed GD video in the Upload GD Video tab.")

# SECTION 5: PERFORMANCE DASHBOARD


elif selected_nav == "Performance Dashboard":
    st.title("Performance Dashboard")
    st.caption("Review your previous interview practice attempts and track your progress.")

    history = st.session_state.get("history", [])

    if not history:
        st.info("No practice attempts recorded yet. Complete an interview practice session to see your performance here.")
    else:
        scores = [int(item.get("Score", 0)) for item in history]
        average_score = sum(scores) / len(scores)
        best_score = max(scores)
        latest_score = scores[-1]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Attempts", len(history))
        with m2:
            st.metric("Average Score", f"{average_score:.1f} / 100")
        with m3:
            st.metric("Best Score", f"{best_score} / 100")

        st.markdown("### Latest Attempt")
        st.metric("Latest Score", f"{latest_score} / 100")

        dashboard_rows = []
        for index, item in enumerate(history, 1):
            dashboard_rows.append({
                "Attempt": index,
                "Date & Time": item.get("Timestamp", ""),
                "Candidate": st.session_state.get("first_name", item.get("Candidate", "Student")),
                "Domain": item.get("Domain", ""),
                "Mode": item.get("Mode", ""),
                "Score": item.get("Score", 0)
            })

        st.dataframe(dashboard_rows, use_container_width=True, hide_index=True)

        st.markdown("### Progress")
        chart_data = pd.DataFrame({"Attempt": range(1, len(scores) + 1), "Score": scores})
        st.line_chart(chart_data.set_index("Attempt"))
