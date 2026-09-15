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
            border-color: #433B86 !important;
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
    # Compact, unified authentication card. Authentication/database logic is preserved;
    # only the presentation is redesigned.
    st.markdown(
        """
        <style>
            .auth-page-title {
                text-align: center;
                margin: 8px 0 22px 0;
            }
            .auth-page-title h1 {
                color: #0F172A !important;
                font-size: 28px !important;
                margin: 0 !important;
                font-weight: 800 !important;
                letter-spacing: -0.5px;
            }
            .auth-page-title p {
                color: #64748B !important;
                font-size: 13px !important;
                margin: 6px 0 0 0 !important;
            }

            /* Style the Streamlit bordered container as one unified card. */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                max-width: 1040px;
                margin: 0 auto !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 18px !important;
                background: #FFFFFF !important;
                box-shadow: 0 16px 45px rgba(15, 23, 42, 0.09) !important;
                overflow: hidden !important;
            }
            .auth-info {
                min-height: 420px;
                padding: 42px 38px;
                border-radius: 12px;
                background: #433B86;
                color: #FFFFFF;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
            }
            .auth-logo-wrap {
                width: 100%;
                max-width: 390px;
                margin: 0 auto 28px auto;
                padding: 14px 14px 10px 14px;
                background: #FFFFFF;
                border-radius: 12px;
                box-sizing: border-box;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 10px 28px rgba(0,0,0,.12);
            }
            .auth-logo-wrap img {
                display: block;
                width: 100%;
                max-height: 250px;
                object-fit: contain;
            }
            .auth-info h2 {
                color: #FFFFFF !important;
                font-size: 30px !important;
                line-height: 1.14 !important;
                margin: 24px 0 13px 0 !important;
                font-weight: 800 !important;
                letter-spacing: -0.6px;
            }
            .auth-info p {
                color: #CBD5E1 !important;
                font-size: 14px !important;
                line-height: 1.65 !important;
                margin: 0 !important;
            }
            .auth-info-footer {
                margin-top: 28px;
                padding-top: 20px;
                border-top: 1px solid rgba(255,255,255,.16);
            }
            .auth-info-footer strong {
                color: #FFFFFF;
                font-size: 13px;
                font-weight: 700;
            }
            .auth-info-footer span {
                display: block;
                color: #CBD5E1;
                font-size: 12px;
                margin-top: 6px;
            }

            .auth-form {
                padding: 20px 34px 18px 22px;
            }
            .auth-form h2 {
                color: #0F172A !important;
                font-size: 25px !important;
                margin: 0 0 5px 0 !important;
                font-weight: 800 !important;
                letter-spacing: -0.3px;
            }
            .auth-kicker {
                color: #64748B;
                font-size: 13px;
                line-height: 1.5;
                margin-bottom: 20px;
            }
            .auth-form [data-testid="stForm"] {
                border: 0 !important;
                padding: 0 !important;
            }
            .auth-form input {
                height: 44px !important;
                border-radius: 8px !important;
                border: 1px solid #CBD5E1 !important;
                background: #FFFFFF !important;
                color: #0F172A !important;
                box-shadow: none !important;
            }
            .auth-form input:focus {
                border-color: #433B86 !important;
                box-shadow: 0 0 0 2px rgba(30,58,138,.10) !important;
            }
            .auth-form label {
                font-size: 13px !important;
                font-weight: 600 !important;
                color: #334155 !important;
            }
            .auth-form .stButton > button,
            .auth-form button[kind="primaryFormSubmit"] {
                height: 44px !important;
                border-radius: 8px !important;
                background: #433B86 !important;
                border: 1px solid #433B86 !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
            }
            .auth-form .stButton > button:hover,
            .auth-form button[kind="primaryFormSubmit"]:hover {
                background: #1E3A8A !important;
                border-color: #433B86 !important;
            }
            .auth-form [data-baseweb="tab-list"] {
                gap: 2px;
                border-bottom: 1px solid #E2E8F0;
                margin-bottom: 20px;
            }
            .auth-form [data-baseweb="tab"] {
                padding: 9px 11px !important;
                color: #64748B !important;
                font-size: 12px !important;
                font-weight: 600 !important;
            }
            .auth-form [aria-selected="true"] {
                color: #0F172A !important;
            }
            .auth-form [data-baseweb="tab-highlight"] {
                background: #433B86 !important;
                height: 2px !important;
            }
            .auth-form [data-testid="stAlert"] {
                border-radius: 8px !important;
                font-size: 12px !important;
            }
            .auth-footer {
                margin-top: 20px;
                padding-top: 13px;
                border-top: 1px solid #E2E8F0;
                color: #94A3B8;
                font-size: 10px;
                text-align: center;
            }
            @media (max-width: 850px) {
                div[data-testid="stVerticalBlockBorderWrapper"] {
                    margin: 0 8px !important;
                    border-radius: 14px !important;
                }
                .auth-info { min-height: auto; padding: 30px 28px; }
                .auth-logo-wrap { max-width: 320px; }
                .auth-form { padding: 24px 28px; }
                .auth-info h2 { font-size: 26px !important; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="auth-page-title">
            <h1>IPER Student Placement Portal</h1>
            <p>Student Login & Career Readiness Hub</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # One bordered container keeps the information and authentication experience
    # visually connected instead of appearing as two separate page sections.
    with st.container(border=True):
        left, right = st.columns([0.92, 1.08], gap="large", vertical_alignment="center")

        with left:
            st.markdown(
                """
                <div class="auth-info">
                    <div>
                        <div class="auth-logo-wrap">
                            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAArIAAATgCAIAAACEjTYRAAEAAElEQVR4nOxdZ4AU1bKu7p68iSAGBBVFFL2AIqJeDCigYEYxe41kECUHRcksGVEkCCYUA5gVJahghKuiYkIRVPAaQGHZnTzTfd6Pb7tes2Fgl9kE9f1Ydoee7tPn1KlTuTSlFAkEgspFIpFwu93XXHPN0qVL3W53IpHo2LHjq6++6vF40nJ/0zQNw8DPYDAYCAQSiYTX612+fHnHjh19Pl8ikTBNU9O0jIyMJUuWdOzYMZlMmqbp9XpxB6VUNBr1+XxEpGlaWkZ1UEEppWlaMpl0uVz488orr3zzzTdN0wwEAmefffaLL76YmZlpWZau6/vzCOefWFMi0jRN1/V33323Xbt2hmGAzxuGkUgk3nrrrY4dO4ICLcvSNE3WV+BEOclRIBDsD9xuNxG1bNkSnFrTtE2bNqVLJiAilglisVhmZqau6zh72rVr98Ybb0SjUSIKBAJKqWAw2KlTp/Xr17tcLk3TLMvKz88nIk3TcJ4J9gfOI/+ss84yTdPlcoXD4Z9++ikzM9M0TcuyLMtK1+NcLpdhGMlkUtO0WCx2/vnnP/XUU36/XymllIIo0KlTp08//dTtdotOKCgRIhYIBFUAHAann346WLPH49m8efOOHTvS+Ahd1+PxuNfrDYfDiUTC5XJZluVyuS688MK5c+cqpSKRCNnH/wUXXLB582aPx2OaZnZ2diwWi8fjkF0E+wNd13Hqa5p29tlnExGErS1btvz999+GYRiGUW5rQXGYpklEMAN4vV7Lsm666abc3FyllGEYbArq2LHjxo0bcbFAUAQiFggEVQDYeFu0aKGUcrvdsVjMMIyXX345XfeHedntdieTyUAg4Ha72VbsdrvvvPPOiRMn6rrudrt1XU8mk6FQqGPHjj/++KPb7YYw4XK55NhIC6CpE1GTJk10XY/FYkTkcrl4udM4z/AXwFAEGjBNs2fPnlOnTjVNMxaLJRIJIopEIhdddNH27dvT9VzBgYSDTixw2s2wf9J7f9wwkUiwYTAejzsviMfjbDYMh8PMMkobLf9vIpFQSlmWpZRKJpP4X36KcoCI4Cfm33FNMpnkL6bznUsBvzgPEi+e+pXLhH2fn9QoKCjAffY6NtyTX40fxIRk2Sg+Qidgrj/kkENatWqVSCQ8Ho9S6u2338b/4m68XimotDR6YwezczyRSAS/KKUGDx48duxYRBhAMti8efMll1zyww8/wGZARIZh4D54BXxYQcSDOeQX5+Xg55bP0s6j5a/jXEwXiuw7clAdTz5/WK9evX/9619erxd2fl5uHOGlTSzuwEsMBxARcRgBr4umaSB+mHnwFbfbbRjGwIEDJ0+ejC+63e5oNLp169Zzzz1369atRAS7EeQVkATZPMT5mniWZVn8OX7h7RCPx/m/cDfeIBgM26gYfEMm1OLz4JzbCuKHzi+STYEHr5NFHZRgGlVKxePx9N4cVI5fotEofxiLxfB7NBplsktxk1gsZlkWqF8pha/wF03T5JHjcXxZPB533h9bzvlJ2l85BTDPGCFGkq7blm9+isA5LTjvU1/v/K+EDb4V8zXQAP9XccRtTJkyxTAM8G5N077//ntcwNSyV5SJ3iKRCK7B/z7xxBMwJOi6DuN206ZNt23bppQqKChIJBLQL/meRWZs/4FhhEIhnEZKqby8PCwrKBnPxUPLTTxFXoFnKS0ocqoxTNNMOgChZ8KECWC8Ho/H5XL99NNP4XA4Bb0pm64wA7inaZrRaBSP4FXm303TDAaDPGPwB+GVn3rqKayy1+uFQ6Fx48a7d+9WSrGKgm/xXlA2zcdiMew4Hi0/mmcVnySTSQzPuUFwN34LCBC8LiVSOzQoHk9F80PnllHppvOahYNOLIBUaBgGC0asEqUF7CaEwI5AbsMwMjMz+Sl+vx+/+Hy+0sKAwanxdb5zkYu9Xi/+5IcahoFtj2MGHxYJHDMMw+Px4G77/76lwTnDiKSDDzVdYc/lm5/icE6OpmmYnBTP5ffCswKBAP7Ecjvv5nK5MIAS39flcnk8Hk3TirwC/4nYsdSDp7LTG7wGRW6LkfANcatatWrhu/xf8EpQWhMTMGOYJRxUHCAJeYWHWu6HYlmJaMKECThQ085ScOQ7ZQIOGgDwp8vlwlowjTEZpOADRMSiG76LuzkvICKfz8dpIyzkERFIFPdHVCn/F4aEO8Bt5BwDz5thGLiJ8+vMai644IJ4PB6JRJRSLNsppbZv3z516tT27dvjnnzx2LFjnapCNBrFqRwMBi3L+uGHH3jFnTuxovkhvoVfNE07//zznQLQwYaDTixQDrsQB/s4z7D9B3Nh7CKn2OH1egOBgK7rXq8XDHqvxzMzDgzS5XLh60Uom39xEr3zXAE3IZuPpFcYKhFut9vv92M8fNiU+Mr7M5IyzU+JwGmHYxjf3et4wOidBz9ek3kQ8+XU64trsrKynOysrBJbmeiNbNEBMg0TDPg+xxg6BQWyZ9X5+vs+vL2CH8qky58UkWDKkanhfItJkyZBBUxhxSkHWCxwGsO0YuBkELLnk3cE/re0RUeoIMsZTG9YUOeBh0ew0O+cRsMwMjIynNPLf+JuvMS6rufk5PCEs0zP9MzDwIu0a9cOrxyPx2FyME3z9ttvz8zM5FfIyMhgCgeldevWDeaoIjP5888/F1m1Imd/BfFDniv8iXzdNBJJzcJBF1tQCYDzz+fzce54ZmYmNp5pmuFwGCFme/VxMhOE687n88EHDGMXb0vcFtQPGyOOB03TTNPkMwMWOd5OFQ2Xy5VIJCKRCLx3LpeLXZ7pQpnmJ8V9MCqn0TVFYh4rWJjqWCwGLmOaJp+7cFGlnmpnPYCCggJWiIufEHs9g8tEb/Ar4y3C4bCmaUopvgmzSMyG2+0GE2dnP6a0fG7+0sAHhmXn0OMXt9uNWbIsC2tdg0IgnUyWPyEiHJaQHhKJRO3atVNPJiiNnfe6riulQOeYqEQiwTIoPonH4yBRpCPiIDdNMxQKEZHP58NkRqNRt9uNoBY4JkAVlmXt3r0bAgf8XHyN89WcI8Qv2AK//vprq1atnnjiiWAwiKejSAbISdO0cDjscrmeeeaZww8/fMWKFbjb7t27kaabYipqND+sYagc6aP6oKKdCGz5ZEMfEyIb68D3s7Ky8OgSn857ALI/f+71etlQzLdlW5/zRKlbty6/IJgRKxaptZN0wTAMvHtWVhbGnEYnQjnmp0SwrR7ArbKzs1M8WrPN3UTk8/mKGCf9fj++7jzpS7sPG0WdigtQRLcrDWWlN7aQsXbIn/ATcVt+rnMAbGhNO9h+y8SMz3k1y11EQasKJ4JeCshhDWLRkFF88ExFbFpAUAIokL0SDOzxIkvMFMv0hrWGPIEbkkMAzcrK4slnRyeHIzDZ4Fvt27dPJBKY1d27dzdq1AhPcRqcsAR4BcMw+MXdbvcrr7wCB4RSKhKJlOZEqGh+SOJEcOCgEwuAigs5ZKUzHo9v2LCB6RJ89pxzzlFKOYOASgOCXyzL6tatG0zc2JPz5s1D2Lxz5Kx88D1N0/zjjz969+6dlZWFXRQIBObOnQsrX3qNqCUC87Bs2TJmAbVq1Wrfvn26Qg7LMT8lAtf06tULUwSBac6cOSkeHY1G2YdaUFDQq1cv8Du32127du0ZM2bs2rULV+51nsPhcJ8+fTBy8PeFCxciHgoBVmpv9FkOeuOp41AyZWcBKKV69OjB3JOInnrqKY7DSiaTzo2TFnCQZjweP/roo/mo0zTtl19+UUqFQiFcGQwGy/2Uyg85NEtCMpns06cPa/Zut/uRRx6JRCKp6YQnnH955plnsDoc2nLZZZe99NJLUO6VUnPmzGnatCmmkb0Vt9xyC68yB3XiSN61axfzEKVUXl7e3XffzUev3++fNWuWUgqZLCWGHOLrl1xyiWanwGCEp5122uLFi3Hb9957b8SIEUxdvGs2btyIC5xhj2rPkMNK4IcScsg46JwISimyvbBKKRi+0nh/The2LOuEE0544IEHlFJerxcHyYcffjhz5kyPxwPKptIT2DweDyxjsO7inIC2l5mZiW9h5GxAg0TPdsXDDz/cNM2CggJcFg6HPR6P3++3LKsSqtexGy8ajdauXVsplZeXhzkv7ZXLhHLMT4lwu92IiCYiMJG9GjPgi8WrZWZmxmIxGGATiUReXl5GRgaC9VBBKPXLgjDgtkDiFozDmqYV8e6XhrLSG9lWDU54g8sAuiz4KQQUGJ93796NfEVowHijvY5q36FpGmLFWc3Fn7quI/QdJ18ikcjIyCiS67sv4P1u2UmDKAKdrvHj/rwBNTufs7idACc03GqoMJhIJHJycmCvSk0nSqlIJIKbb968+a677oKrCB6iSZMmvfrqq1dccQXmMBaL9ezZ87vvvhs+fDgeDVJ86qmnZsyYwQH84IGw8NeqVQs2dpfLFY/Hc3JywuEw2WIHnDjJZNLr9Sq7FgIR4UMiisfjmqatW7du+fLlLKkYhjF+/Pj//ve/119/PcjmrLPOGj9+/LZt20466SRd19kldNddd2F1wL54HtgSRhXPD+FQgLsE94e/Jl10UrNw0IkFTlZbxLzMia1kZ9MqR56rUgoKHNkpuZyDhK/wL5qm7dq1C6xn5MiRp59+uulILJ46dSoRJRIJwzCslAXJQb6oZg8JF1I5ORiQZVnM48Bc2NHIPkvsE03TMGwqV+o57gZtgxy51ORIBMf5ajoy+MF6du3a5WSX6fIjlHV+SoRSis3vys7MTu3OB+9QSvl8PnzdcmRy87Q7Y+xLvA8OWh6n82IegGVZexVbNU3bvXs3OOD9999/xhlnYKFLpDdyWGh5hEwwGE8kEsFZwrPH9lhrPwr4lwZ4uMl2jZO9+xBqh08wwv0JOeRhp1cNKO4C0OyUAYDPe1zDMhmuD4VC7PlOQSea7W9KJBITJ06E7x8n95133jlkyBDsLLL9AvjWyJEjly5dCkET3GnUqFG7du0CMWBISilnYgI5vDlEhOgTBNBgnCCDIo4ziAsvvfQShDkwoh49egwaNIiPf/aD1KlT54033jjkkEOYGleuXPn999/TnjuiyCRTBfNDXjLNjlFIF5uqiTjoxIIUAAUjOAt8EJImwuXIjjAH5fEJxzsE+x9mqNq1axcUFEDhe/zxx5PJJI4QpdT27dsXLVqEcJ4aFELFbMgwjFAoBJkdiiYqrSL9GvpKekPSBKkBpTMcDiPqcO7cuXAJ12h6EwCQOxEtmEgkdu3a9dRTT0FsAvOB/8t55hmGEY1GdV2Px+OXX3457Pa6rvv9/mAweNddd3k8HuzlNB57mqZ9+eWXqNAAMejf//43hziQHf1HRIFA4Kijjrr++utxfsMC9NFHH4HTpms8gv2BiAX/D8jySOUCfcMBBmEZkjLKyycSCewr2DwhdcLk5fV6YfzMysqCy+rkk0/u2bMnqB+WwzfeeAN2uZpVcx5ZUkSUkZERCATgmGRnQSAQgLUZ3sqqHuzBAo5I9/l8WI5TTjnl5ptvPgDoTQBYlpWRkQGZ++mnn4alATpM69atmzdvTkTs3IH/CMSA6Lw+ffocf/zxZJcyfP3113fu3Ak/UXrF959++gk3hMjSpk0bONogDXAwARHpun7WWWeBbRqG4fF4vvjii4PZaF/dIGLB/4Nrb5G9zWAVh1SrlLIsC+Xl3W43NmosFnO73V6vF7IwzOnInCFbmY7FYoMGDYJFHZbAV155JRwOQ7CoKWBrOeLXiCiRSPj9fkyX6WgE5/f7Jeen0qA7CiT4/X5EKgwZMqSm05sAYHM3VvmTTz6pU6cOG73hMLLs+sr4CixDcHVFIpH69et3794d13g8HsuyZs6cuddohrIC4TWIR8EnRx55JBH5fD7wUsgKZCdnHnbYYZpduTkej//222+GYbAfVlC1ELGgEPBlYtvAj4BQILYW4OTD74jrxv+adv1wFBiBZMCGdNgPjjrqqM6dO6MmKBEppWbMmAGFu2rfet+haVowGOSMJpTxR4Y0OWKswGjSW3ZekAK6XTUBf0KKPf7442s6vQkALJlut2HMz8/fuXMnjJREVKdOHUS/8spCBYcrIR6Po6XyFVdcQXZbxXg8Pnv2bHCz9A71kEMOQTgLbKum3bKhCMAikGDCuZQY/0Hry69uELGgEIixgqUOqd6WXRUckQREhIPfNE2PxwPfLWRkGHLJrhACfdq5VzVNu+2228jO5U0kEqtXr6bKalmUFpimGQgEOPMKMQQc4M3uQ1wsPsJKA4gQfoRoNArzQDwer+n0JgCU3foIzIS3GwIDuVQffsJVhCQF0+6AZVlWw4YNL7nkEtxK1/W8vLyvvvoqvfVaDMOoX78+HoHbbty4EXYO0CdIFKKApmnffvst2bYQn89Xu3Zt1jEEVQ4RCwrBKViWZcFHi2Nv8uTJPXr00HU9IyMDhTi8Xu+4ceNmzZoViUSwXWFmQFwhEaE6WzAYxE2Q9gNPG8zvtWrVevfdd/Pz86v6pcsAyPWJROLhhx/u3bs3wp0yMzMNw+jZs+eHH34Yj8fxvrCpVPV4DxZw4HQsFoP3CrXtajq9CQDI36FQCHLA8ccfDx8BlOyff/45Ho8j2gBpBZx0wOICEjqOPfZYJhWXy/Xmm29SuotYN23alG+olHruuefIEZAEhywLIh999BGcC7BpnXzyyWLKqj4QsaAQONqhCsNl/vDDD3u93mHDhj3++OPgsJZloUro6NGj77777jp16nTt2hW1Ntn5By9DLBZDLRHOuapbt+7RRx8Nh18wGCSin3/+Oe25XhWK3NzcrKysgQMHLly4EN4WnDrz5s275JJLpk6divTlSCTi7GgiqFBAKoWEik9AdQcAvQnI7p6MSCalVP369aGOwzK0atUqhDpBIGBRQLMrH8NasHXr1kceeSSRSGRlZaHk0caNGymt1iOl1Iknnsh1FZGvSHbQK9IgmUNu27btlVdeQc4CyPWcc84JBAJizaomOBjZhCpW9o4/RzxBXl5ehw4d+vfvz8VAnAUGOG5AKfXFF18wxUPa9Xq9iPdxhhfg/scddxzOUVgFf/zxx+oZmgdfAFqjElEymdy1a1fHjh1HjhzJnmxYIzW7AUx+fv7IkSP9fn/v3r0hEzhnlfaBAXGJCL4S027aXVAxGImbKwLUn+HALud/VQK98bqQTTbYBUlHG/t169ZNnz69devWyFxv2rTp2LFjX331VbKjz3jYml2Fhl8NlnDObUPMClca5rdzDqAI0htYVyWAKR785JJLLoGLk4ji8fjmzZtXr16t7G6EfD1XJkB61BVXXIGooIKCApzNyBpIXS+hTLAs66KLLkLVZNhHN23aNHfuXDgcYcPgmkX9+/cnR025E0888bzzznNyYw5R4uAYPAXrDt+ls9aIc672/10EB6NYUCJAkSiDde65537wwQeotsY1eRBhwH13YD/48ccflV2kEwwasXjYmWzFJaJwOMziAiLGf/vtt2poN+NAIZQ/g0vl3HPPXblyJbIqUAYH3haoqkjQwJ7/9NNP2UBSJuBbXOwP8ckIbIYBBsdAIBCQ5HsnmN6gRKJsBlU8vbFsjT8hJiLwFiw7mUwuW7asbdu2Z5555sCBA0EYhmFs3LhxwoQJnTt3btGixbvvvpvi/vgFzim32717927Yz+GYU0rBZMVb0jAM2EXIFh8PgNBXVM/kolLNmzc/44wzOC2IiLp3756fnw8ehaPXKfwlEonJkyd/9913EBQQdUhEeXl5UHjSOM66detef/31vPqmaY4YMWLbtm1EBHEEBq2RI0e+9tprqAEDkhg2bBgsXviEBQhd130+H3skOaIZbKemS3vVGQejWFBEO3H+bhhGy5Ytf/zxR66q5vP5br/99rfeequgoCCZTP7222+zZs265ppriAilTHfs2IEDkhw16WBO58dFo1G/3799+3YkQIJ74mflvvrewTEEqNqk6/o555zz3XffIXkafPnWW29dsWIFWPP//ve/CRMmXHbZZdBWP/vsM2es5b6Dy9uheOpPP/3Ut2/fJUuWJJNJv98PsYBtFVIuicEFX/EnTutIJFLR9GbZTQ6xHJAmWTLWNG3kyJGXX375Rx99RHZkHEt4yOL57rvvHnzwwdLuD8GaHHWLv/nmm8mTJ7dr1w7OchyEt91229ixY//44w9cnJmZCcnAWRevpoN987AZLFy4EHH+iDbYsmVLu3bttm/fTnZCECvoRHTNNdcMHz4ckdFcqM0wjG3btiml4H1Ii0EFVDd+/PhDDz0UNh5d13ft2nXdddcppWBF+OOPP26//fbc3FzIDdCmrrrqqltuuSUSiaB8IforQhhC5AF6l3PbApg3wHyqp7X1AMDBKBYUB+wEuq4PGjRox44dqEdERA0bNly5cuXDDz980UUXwQR36KGH9urVa/HixWvWrEFU18qVK71eLxQy9qtBpYa+iwjEP/74Y8uWLWRr4aZpcvnP6gYUYwCvHzJkyFdffYXjJBQKNWjQ4KuvvlqwYAFKmCWTyXr16vXv3//ZZ59dsWLFOeec43K51q1bV47jh6uZEtHYsWNPPPHERYsWrVq1iqeIE/RJTIV7AskyoDccmX6/v6LpjY3POFRQuoOIUEGvffv2ubm5kBRhM2MvG8fhJpPJ3bt3p7g/hAlN0x599NEGDRq0bdt2xIgRH374IS6AAv3ss8+OGjXq2GOP7dy58+bNm8nO2vf7/dXQFFc+YAI5l++4444bNWoUljszM9OyrG+++aZJkyYjR45USqGybygUmjJlSiAQeO2117KystgECDkANcf4cE2L2o2CZjk5OUOGDGG1XtO0zz77rHnz5n/99deUKVP+9a9/oRgzZ3g1a9bskUceISK/3493hFYACQCSHxdUBUkgCwOPOGCWuLpBxIL/Nx6sX7/+wQcf5B4htWvXfuutt/797397PJ5oNIoOIrhS1/Wzzz575cqV999///bt2yEpwyzmzNa17KZEsVjsyy+/hEOBy3q0bNmyGho5oVgQkWEYr7322uzZs7Fjiejwww9//fXXmzZtCl8183oY+tq2bfvOO++MHTv2yy+/LEdoG4tQLVq0GDVqFHjWY4899vvvv8fjcWi6GEaJVdMPWjA5EZFydIupaHrjSh445sn28ScSiQsvvHD16tU4zOAb8vv948ePX7t2LXxDyWRy2rRplLLHgWVZBQUFv/322+mnn96rV6+///4btmU4DiCFkx3ckEwm16xZ07hx4xdeeAHKJcipeordZQKqhGGiIKzn5+ffd999d9xxB0qJEBGSAMeNG5eVlYXkoOzs7KFDh0JRyc/P1zRt6NChDRs25KOUS7Sla5y4rWmaffr0ady4MfyMKKz5448/Nm7ceMiQIcFgMBgMcnvl008/fcWKFbVq1YrH41yTkYggsqBcLAtDRBSPx3fu3IlEJ1gQJbS5gnDwigVFjhbDMO677z4WP0Oh0KhRoxo3bozIO5/PBw9uNBqFVgSH34gRIzp37gw+hS+CXULC9fl80LndbveSJUuwt6HK1KlT56STTqqg1vX7A6Rl7t69W9O0adOmhcNhvIuu6/36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zzd//nwQYTKZRMRDJUjGZR2PZVmRSEQ5hKf0mvR4v+C2zvScsqKs9JyW9a24cXKyEtbrv//9L7QIloNPO+20sWPHIphGKfW///1v6tSpzZo148AICDpHH330999/n/rcYcKzLOuPP/7o06dPVlaWbrub586di6ekWJrqRudgCJZldevWDbZqCGHz58//+++/MVScVshHwCd8AGHyy/30MhtDEKXpJAvWNSGrYuEDgUBeXl5GRgY0WrilEWgaiURSGGMRoW2aJqLo8Tgc2PjfaDSak5ODcGL0DgecN1FlT0nAsPEKOF91XY9GozAeWJaVn5+fmZkZDoeTyWQgEMCkG4aBXzRNS51hkS4oO/8iIyPDNM2pU6feeuutiNqNx+OIkr3yyis3bdo0bty48847D1NhGEa3bt3ee++9RYsWZWVlKaWSyeSPP/44atQoTDUbeJRSiUSCzVMsFem6DoGJzWtKqezs7OKTXzlQSiEVE0wE0fvYIWkZj9vtxswQEbhA+ZSt0saJ+YR2pZRyu91btmx5+OGHcY3f749EIvfee++iRYvAy3Rdv+qqq1avXn3sscfizqFQaMmSJUU8oEQUi8UCgYBzXZhTQHpIJBIYTDwe9/v9uENp68hpKdh0wWAQMxONRtngwRbscjhZyoqyjkfTNJ/PB4InIsSlp5FiNU1DRD2GoaX0kKZAWek5XeubxnGWOFTYBkzTvP7660OhEMjeMIxLL7107dq19913HxJqiKh+/fr9+/ffsGHD9OnTEUyDe27fvv2mm24q4p1xnjtE5PV6IYJomnb44Ycnk0mksBFROBz2eDx+v59T2EpEdaNzj8eDXCSkJUPfxjavW7euruvgSLCU8MxgnJgTzc4ZKQfKGRrjDNrU7QRFZnBEhKQgGKg1TYvFYtg5Sins0tJubthwJi8481KczgielyIsm084IoLmRHvWJMBlyIrBzcF58QrOjY28FM0R2cD+ZrwyP70SaAWvFo1GQQQTJky49957na+cTCanTJmyePHixo0bs6BGdl4QEV133XWrV68me2Iff/zxLVu2sKiHtJ8iMZ60Z8wz1zwAD6oqJ4JmJ1JyNn8kEkkRaVxWOGeP6accPsXSxklESGzjBXrsscf4W4lEon379iNGjEgmk2x01TQtKyura9eumu3dXLJkCQ8Pnyil2MPldPdik7KPCdmtnJHL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glNOOWXhwoVoDUdEqPcMGt61a5dhGEhgq4QhVSg0OysVa4FoiZ49ezZq1MgwjMzMTKhGvXv3JqJgMOj0plWHcTZr1gyqeUFBARHddddd4NVKKXbYsa8qxXGu2fWawLuICGExU6ZM+eWXX3hz4ZoUrw+3ERGhfAKGyomOXq+3T58+f/75J/S6NM5PWYHD6LTTTnv66afz8vLINigmEgkEXiB4roiWUt2QTrGAiBChCs47evToHj16IGAexdq2bNly+eWX79y5E+cNaA56A84PSMq6rjdr1mzr1q39+vXDLKeoYsbSALO5Cy+8EFG+pl3254knnkAUHgI3yDbQwYTAti/Ndt0FAoFXXnllxowZVVg9je0Elp1wD+552223nXDCCSxFLVy48KeffmJTATsLzH1oRYMd3rJlS+w0InK5XBMnToSe+tBDD2G6EonEP//8c8MNNyildu7cyZmQezWGsxmQzyfoSRdeeOH06dOdIlelVaRhd5Vlp3p7PJ5nnnkGMd5oML18+fJXX30VJlBd11ELK2FXwq7oEbKhAn8qpbKysvb6XKeDGcvKZp5YLPbAAw8899xzXFueDeyVJhNghD6fDwRwyy23TJ8+HTY5RIDCYle7dm1I59UwVqAc0Bx+fbBBv98/efLkjIyMvLw87J1Vq1Y988wzSL2mPX1AVFnHRonjHD16NKy5KMq5bNmy559/HrzasFt26Xbz3xQ3j8ViCKQnogEDBjRu3Bgs3TTNa665BjZFIkJqVQp+YhgGxAKIJrpd7RFu38mTJ7/11ltI4bHsFJWqAozcN91007Rp04goEokEAgHDMHbu3GlZFszniCKsnpW4KI1iAc54+L2UUjiYp0+fDpdSQUEBzCkbN2684oordu3ahW9lZWXhYIbXis9m+Ic4QyaFtUBzlLVClOxpp5129NFHoxwYJNDnn3+egwnAccDlOeAALYBxcWZm5sqVKzt37lzlopxhVzXW7JRL/Dl+/HiWZnRdHz58OIKw+CxBnDPtg30YosbRRx9Ntl193bp1nANmGAZEe1bokUDPj05B1spu5GjazQU46CYQCPTs2fOWW26BhRyh2vs/XfsC3e7mDJM1vP5nn332dddd53a7//nnHxDwoEGDTNPkhCiyxaxKOK40TeNNQXadnxT0r4qF7gIwzCaTyUceeQQJ01yhvUrAvT0havft27db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" alt="Institute of Professional Education and Research Bhopal logo">
                        </div>
                        <h2>Prepare for<br>your next opportunity.</h2>
                        <p>A focused career-readiness platform for students preparing for interviews, group discussions and campus placements.</p>
                    </div>
                    <div class="auth-info-footer">
                        <strong>Career Readiness Hub</strong>
                        <span>Practice. Improve. Prepare with confidence.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            st.markdown(
                '<h2>Welcome back</h2>'
                '<div class="auth-kicker">Sign in to continue to your student placement dashboard.</div>',
                unsafe_allow_html=True,
            )

            login_tab, signup_tab, reset_tab = st.tabs(["Student Sign In", "Create Account", "Forgot Password"])

            with login_tab:
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
                st.markdown('<div class="auth-kicker">Use your registered official email ID and Scholar ID to set a new password.</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="auth-kicker">Registration is limited to official @iper.ac.in email IDs.</div>', unsafe_allow_html=True)
                with st.form("student_signup_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        first_name = st.text_input("First Name")
                        scholar_id = st.text_input("Scholar ID")
                        email = st.text_input("Email ID", placeholder="yourname@iper.ac.in")
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

            st.markdown('<div class="auth-footer">IPER Student Placement & Career Readiness Portal</div></div>', unsafe_allow_html=True)


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
    f"<div style='padding:8px 0 2px 0; font-size:18px; font-weight:600; color:#0F172A;'>Welcome, {st.session_state.get("first_name", "Student")}</div>",
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
