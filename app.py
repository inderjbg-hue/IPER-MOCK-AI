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

# Persistent production account storage.
# Configure DATABASE_URL (or POSTGRES_URL) in Streamlit Secrets.
# Hosted Streamlit containers should not be used as permanent SQLite storage.
DATABASE_PATH = os.getenv("DATABASE_PATH", "students.db")
DATABASE_URL = (
    os.getenv("DATABASE_URL", "").strip()
    or os.getenv("POSTGRES_URL", "").strip()
)
if not DATABASE_URL:
    try:
        DATABASE_URL = str(st.secrets.get("DATABASE_URL", "") or "").strip()
    except Exception:
        DATABASE_URL = ""
if not DATABASE_URL:
    try:
        DATABASE_URL = str(st.secrets.get("POSTGRES_URL", "") or "").strip()
    except Exception:
        pass

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
        # SQLite PRAGMA statements are not supported by PostgreSQL.
        # Translate PRAGMA table_info(<table>) into information_schema.columns
        # so schema migrations work with the persistent PostgreSQL database too.
        pragma_match = re.match(r"\s*pragma\s+table_info\(([^)]+)\)\s*;?\s*$", sql, flags=re.I)
        if pragma_match:
            table_name = pragma_match.group(1).strip().strip("\"`").strip("'")
            cur = self.conn.cursor()
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
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
                duration_seconds REAL DEFAULT 0,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
        """)
        # Backward-compatible migration for practice duration tracking.
        interview_columns = [row[0] if USING_POSTGRES else row[1]
                             for row in conn.execute("PRAGMA table_info(interview_attempts)").fetchall()]
        if "duration_seconds" not in interview_columns:
            conn.execute("ALTER TABLE interview_attempts ADD COLUMN duration_seconds REAL DEFAULT 0")

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
                duration_seconds REAL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        gd_assessment_columns = [row[0] if USING_POSTGRES else row[1]
                                  for row in conn.execute("PRAGMA table_info(gd_assessments)").fetchall()]
        if "duration_seconds" not in gd_assessment_columns:
            conn.execute("ALTER TABLE gd_assessments ADD COLUMN duration_seconds REAL DEFAULT 0")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS student_profiles (
                student_id INTEGER PRIMARY KEY,
                tenth_percentage TEXT,
                twelfth_percentage TEXT,
                undergraduation TEXT,
                post_graduation TEXT,
                internship TEXT,
                certification_courses TEXT,
                activities_participated TEXT,
                achievements TEXT,
                professional_interest TEXT,
                about_yourself TEXT,
                profile_hash TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
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


def create_student(full_name, scholar_id, email, password):
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
                full_name.strip(), "", scholar_id, email,
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
            SELECT timestamp, domain, score, mode, question, duration_seconds
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
            "DurationSeconds": float(row["duration_seconds"] or 0),
        }
        for row in rows
    ]


def save_student_attempt(student_id, domain, score, mode, question, duration_seconds=0):
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO interview_attempts
            (student_id, timestamp, domain, score, mode, question, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                domain, int(score), mode, question, float(duration_seconds or 0),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_student_profile(student_id):
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM student_profiles WHERE student_id = ?", (student_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else {}


def save_student_profile(student_id, profile, about_yourself, profile_hash):
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO student_profiles
            (student_id, tenth_percentage, twelfth_percentage, undergraduation, post_graduation,
             internship, certification_courses, activities_participated, achievements,
             professional_interest, about_yourself, profile_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                tenth_percentage=excluded.tenth_percentage,
                twelfth_percentage=excluded.twelfth_percentage,
                undergraduation=excluded.undergraduation,
                post_graduation=excluded.post_graduation,
                internship=excluded.internship,
                certification_courses=excluded.certification_courses,
                activities_participated=excluded.activities_participated,
                achievements=excluded.achievements,
                professional_interest=excluded.professional_interest,
                about_yourself=excluded.about_yourself,
                profile_hash=excluded.profile_hash,
                updated_at=excluded.updated_at
        """, (
            student_id, profile.get('tenth_percentage',''), profile.get('twelfth_percentage',''),
            profile.get('undergraduation',''), profile.get('post_graduation',''), profile.get('internship',''),
            profile.get('certification_courses',''), profile.get('activities_participated',''),
            profile.get('achievements',''), profile.get('professional_interest',''), about_yourself,
            profile_hash, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
    finally:
        conn.close()


def generate_about_yourself(profile):
    if not client:
        return "GROQ API Key is missing. Add GROQ_API_KEY in Streamlit App Settings → Secrets to generate your personalized About Yourself response."
    prompt = f"""Create a professional, natural and interview-ready 'About Yourself' response for an MBA placement student at IPER Bhopal.
Use ONLY the information supplied below. Do not invent companies, roles, percentages, achievements, skills or experience.
Write in first person, confident but not exaggerated, suitable for a 60-90 second interview answer.
Integrate education, internship, certifications, activities, achievements and professional interests into one coherent story.
If some fields are blank, do not mention them. Avoid sounding like a resume being read aloud.
The response should be easy to speak naturally and should end with a forward-looking statement about the kind of professional opportunity the student seeks.

Student profile:
10th Percentage: {profile.get('tenth_percentage','')}
12th Percentage: {profile.get('twelfth_percentage','')}
Undergraduation: {profile.get('undergraduation','')}
Post Graduation: {profile.get('post_graduation','')}
Internship: {profile.get('internship','')}
Certification Courses: {profile.get('certification_courses','')}
Activities Participated: {profile.get('activities_participated','')}
Achievements: {profile.get('achievements','')}
Professional Interest: {profile.get('professional_interest','')}
"""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role":"system","content":"You are an expert MBA placement interview coach. Produce polished spoken English while staying strictly factual to the student's supplied profile."},
                {"role":"user","content":prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.25
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"AI generation could not be completed: {exc}"


def database_status_message():
    if DATABASE_URL:
        return "Persistent student database: PostgreSQL"
    return "WARNING: Local SQLite storage is active. Configure DATABASE_URL in Streamlit Secrets for permanent student accounts."

def require_persistent_database_for_production():
    if not DATABASE_URL:
        st.warning(
            "Student accounts are currently stored in local SQLite storage. "
            "On a hosted deployment, this storage may disappear after a restart or rebuild. "
            "Add a persistent PostgreSQL DATABASE_URL to Streamlit Secrets and redeploy "
            "so students create their account only once."
        )


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
    duration_seconds = 0.0
    try:
        ends = [float(seg.get("end", 0) or 0) for seg in (segments or []) if isinstance(seg, dict)]
        starts = [float(seg.get("start", 0) or 0) for seg in (segments or []) if isinstance(seg, dict)]
        if ends:
            duration_seconds = max(0.0, max(ends) - (min(starts) if starts else 0.0))
    except Exception:
        duration_seconds = 0.0
    conn.execute("""
        INSERT INTO gd_assessments
        (scholar_id, topic, participant_names, transcript_json, report_json, video_filename, duration_seconds, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scholar_id, topic,
        json.dumps(participants, ensure_ascii=False),
        json.dumps(segments, ensure_ascii=False),
        json.dumps(report, ensure_ascii=False),
        video_filename,
        duration_seconds,
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
                max-width: 1120px;
                margin: 0 auto !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 18px !important;
                background: #FFFFFF !important;
                box-shadow: 0 16px 45px rgba(15, 23, 42, 0.09) !important;
                overflow: visible !important;
            }
            .auth-info {
                min-height: 500px;
                padding: 24px 28px;
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
                max-width: 185px;
                margin: 0 auto 18px auto;
                padding: 7px 7px 5px 7px;
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
                height: auto;
                max-height: none;
                object-fit: contain;
            }
            .auth-info h2 {
                color: #FFFFFF !important;
                font-size: 24px !important;
                line-height: 1.14 !important;
                margin: 10px 0 8px 0 !important;
                font-weight: 800 !important;
                letter-spacing: -0.6px;
            }
            .auth-info p {
                color: #CBD5E1 !important;
                font-size: 13px !important;
                line-height: 1.55 !important;
                margin: 0 !important;
            }
            .auth-info-footer {
                margin-top: 20px;
                padding-top: 16px;
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
                padding: 18px 30px 18px 18px;
            }
            .auth-form h2 {
                color: #0F172A !important;
                font-size: 28px !important;
                margin: 0 0 6px 0 !important;
                font-weight: 800 !important;
                letter-spacing: -0.3px;
            }
            .auth-kicker {
                color: #64748B;
                font-size: 13px;
                line-height: 1.5;
                margin-bottom: 16px;
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
                padding: 8px 10px !important;
                color: #64748B !important;
                font-size: 12px !important;
                font-weight: 650 !important;
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
                .auth-info { min-height: auto; padding: 28px 24px; }
                .auth-logo-wrap { max-width: 185px; }
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
                            <img src="data:image/png;base64,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lStXhmCq0Z49e+ASqoGqZBTlHcXj7+zZswsXLhw2bNgdd9wxZsyYZcuWYdSKvTSQErHxCliiFc84BH4gXNUoVGyBQVCWZcXExNSsWRO9OHv37t20aRN6klVCAG4oxtiTTz6J/gAp5Z49e1TKzkUxPdJP8+bN8Y4nTpyYPXu2WsNISFOnTj148CCmAbVq1apo0aL2WCMVyQbn18QlPfuaZYUQe4vD4Zg9e/Ydd9yxc+dOnP6KFSs2atSoTp06WNKAc37ixAmwRYmosmtorMcMF7zy6dOnT506ddkWkLoLRmuMGjXqiSeewJgQt9tds2bNevXqVatWLSYm5uDBgzNmzNi+fXvu9xjSgKZpx48fx5YDeLopH+DVvhLQtaBiz9R4ok0DdYU8qH32Gjj2NePxeF555ZX69ev37Nnzgw8+mDt37muvvXb77bc3a9bsp59+yjIKFuF0OhljiYmJW7Zs2bRp019//XXkyBFV/s/r9eIiVLpI4VHF8Add1/v376/Ktzz11FMnT550Op2Yv4Z7kHP++uuvz58/H53/jLFTp05lGc6Xy/G/9dZbK1SogHtzwoQJmzdvxloaDofj66+/fvnllw3D8Pl8JUqUGDx4sD1iTbX6sf8y5I2IFa5BSlCCg67rf/7557333otVU0qXLv3ee+/99ttvGzdu3LJly9KlS8ePHx8XF7djxw7UADC83R6ugJGgBw8etHclvJwbD8nJ5XJNnjz5rbfeMgzDsqxWrVp9+eWXf/311+bNmzds2PDFF1889NBDALB169bcUwIeOi+++GLPnj137NiB2b8qi/hql5uUoAq26ICjR4+Gi+p5UEEU2ahs27179/bs2fPZZ589dOiQXdnKyMj4559/5syZE37iqPTaL7/8csiQIR06dKhfv37jxo0bNmzYtm3bAQMGfP7555qmuVwuXL1gSzgvJCOML+X3+2+99da77747EAi43e6tW7feeuutixcvdjqd2M1tw4YN99xzz+jRoyHY6E3X9eTkZGSFi3WGoVpftWrVRx55BAc/LS2tf//+u3btOnHixKhRo+6///709HR0Nvzf//3fddddp9we9lLqyAdo6Lvmram5h34NvxtuOY/H89hjj+GmKlas2Pfff9+4cWMsZWEYRpMmTZo0adK1a9cPPvhg165d9erVU8HjqpQC53z79u2nT5/GvJ6SJUsWLVr0snViQUpwOBy//vrruHHjsCbBTTfd9P3332MxCb/fHxMT06VLly5dutx0001bt27NyMhAL0gOl0VH+tGjR++4445ff/0VAJYuXVqvXj113l0bTQVU1xdUrfx+P7Im8kGNGjXyJh7iNVE+wCiGAwcOdOnSZceOHSggx8XFDRs2rEmTJjExMUePHn344YdD6vyoCiL79u0bNGjQTz/9FHKL3bt37969+6uvvvr8888//PDDcuXKYWx0IdxiSGw+n2/ixIn79u1btWqV0+ncsGFDz549mzdvjjLWli1bUB2vUqVKuXLlVq1adYnxGmi5Gjp06CeffLJ161an07lt27bOnTsbhrFjxw485YUQb7755sCBA9F7gcZkVBQwQwgzB9ELohQd0hWuTVZQgoBhGF9//fWaNWswnPnpp59u3LhxRkYGlkMxTRPDURo0aDBp0iRMo0eTLtpSMKrH6XTOmTNH7eTrrrsOVeAsEyAL6F1wyyUlJWFy7KuvvhoVFYXh3sheeDb17ds3MTExuwDw85REzoUQLpdL+Ug+/vjje++9t1ixYkqSyjL1F3LM58pNFfvLDCyZiTrQ7t27N2zYoGma3+8vUqRI3bp180w2yDd+v98wjJSUlIEDB+7YscPtdns8nkaNGn3yySf2i0dGRh48eDAkWwoAFi9e/OKLLx4/ftzlcmF1nYYNG0ZHR6O4fejQIU3Tli5d2qlTp7lz56LAG9Ly4QqbGjhXdX855/Hx8XPnzn3wwQe/++47fMcQtmvRosXMmTMXLVr0888/M8aKFy+u9O887As81t944437778/MTFR0zR0FGHl18qVK48ZM2bw4MFoYlJpibi8DcOIjIw8fPjw/Pnzp02bZl+0RAzXctcdXdfPnj377rvvohGmatWqffv2xaMQV7Ou61huxTRNLJusSgErN6zT6VyzZs0nn3yi6uW2b98ebHUaChqoKKxevXrx4sV4fN9+++1NmzbFGHCVdoteNZ/PV7x48ZDyANltKtM04+Pjhw4diu+1ffv2BQsWqIpgOfRGRuLE/4YAPYqFpEAYlvJXBe+klGPHjj127BiWP7vxxhvzrPMpdwV+/b333vvpp59QSyhVqtScOXPq1q2LnRswz/b2228fPny4XTrGL86fP//48eMOh8Pr9dauXXvWrFm//fbbzz//vHjx4mXLlk2cOBErUG3btq1Xr1779u2zW8MLA1D0VmY0v99frFixr7/++p133mndurUqoVG6dOmuXbt+9NFHS5curVKlCtY3lVJWqVIlNjZWraiLJSQUzjp37vzEE08gLWHGIl68efPmLVq0OHLkCJYgxP8ahuFyuZxO5+HDh8ePH9+uXbuHH354//79yoRFlHAN6gqqiice+ps2bdq0aRNu1969excvXlwlLuF/1emmSgappYwXWbhw4eDBg1NTU1HbaNGixa233qoyZS7DGsJbLF++XFVvve++++zZanjkQbD0W+6fCjmma9eur7766r59+zBsY9CgQagGqdxvpS7gWKWkpHg8HrvrxU6QqLIkJSXZBcnLLA3gYR2Sgp6YmDh69OhvvvkG9T8AePDBB/FYyQPHKz5wuVzHjx//9NNPcbiEEOPHj69cuTKWCMURtvcYCNG0NE1DhaN9+/azZs0qXrw4VmeRUtasWbNmzZo1atTo06ePEGL79u0vvfTSxx9/rOzjhURXwOWBsUC4JBwOx6OPPjpw4MADBw54PB4ASEhIKFWqlMvlwhi/w4cP49dr164dFRWVyxWLY6j4A/9pGMbnn3+OzjYcc5xQTdPmzZu3ePHiokWLxsTE4PziHseZwlwf9RZqqVzBPCRihQI8FOyO4t9//x2CbtWmTZtCMCQpRChQp9uHH37o8XiqV6+OUUkLFy6cP3++EAJN0tHR0ZMmTbJ34LkMr4PV5//44w98kTJlytSoUUOF4YcLWRdVc0IIUapUqZEjRz766KOc840bN77//vvDhg1TXcMcDsf27dvT09PRxNGkSZPffvutR48eqKmE7yLcdco5cZm3Gap3uq7v3LlzxowZN9xwAwalrF69et68eXv27MGmmx6P54EHHujWrRvaCfOmK2B1B4fDsWrVqj179iAFli9f/t5771WVveF8t3bIUOCtA4FArVq1pk+fXrx4cb/fr7gEuaFLly5Dhgx56623dF1fsGDBxo0bGzdunJ1l70qZakN+wFmIioqqXbu2/UzHVbRnz55Vq1bhmLRr1w4NQXZRLId9bXcU49YYP378888/jyutdevWQ4cOXbly5fr167ds2YLl3HNuwsEYu/nmm++7777JkycrhiBv8zXoV8BFg9sG118gEIiNjc0ujQVVBPw5NTX16aeftv8Vuzh5PJ6YmJiZM2c2bdoUz52QAOcCEt9w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avWDO+XaKOJCrJBH46Zd+27WrJk6JX///Xd7kTu7Lm/fwxjhg7+ZPXv2X3/9ZTfyKosB/hMXd2YLbF1/6623sERXOP3kbTniAVeqVCmsRYHx9Ug8mDKmnhxXPHajdDgcWKYtPT0997KPEKJt27b33Xef3++PiIjYtGnTW2+9pawWlycSN19kApT9IyIi7r33XgyfT05Onjp1qn2FIB+oqbcXr8ZWrBMmTIDzO5KGNOFS1o86deoofWL16tVgC6NUUVjqyowxjIuzF+lTDWqUSIuLCj+/ePFidan69etj1H9hsHLgMb179+5HH300KSnJ6XSuX7++Q4cOnTp1ateuXYcOHW677bbHH398586dGK3g9/sfeuihjz76KCoqSgVV5/JF7BHYycnJSsPACrh2nQxtVuheQvLGDlqXrRgBsUKh8yuoDYa/uemmm4oUKYIxfytXrsRcrSyN0VjZ4siRI++88067du28Xq+maRkZGS+99JJqwmMvhoy/xNuhIX716tVTp05Vemt+naFoG8VmcHhAf/rpp2Dzcqs8KTybIiMjz5w5M3DgQCxOmek+ulAgkDoZn3/++TJlymRkZGia9u677x46dAhDlRT5Ff41oOwADzzwQJkyZTwej8vlmjlzJvrSlVNE5TShOwRdx0jDY8aMOXnypCqWnuWaUWPepEkTrD4CAN9++63P5wvJ91anlaZpv/7666effoqqibqavQmgMgOqZKuvvvoKJZuIiAgM8C8ks4AjUKFChVGjRqG84nA4Dh8+vHTp0pUrV+7YsQO1IowBq169+hdffDFlyhSXy4UWWntP3IsyDtuL2CvNSelzSiJ0uVxnzpz57bff1qxZY1lWyJgT/kO6gjKpoyBfrVq1IUOG4Ll/9uzZ8ePHh4QkQbBvu67rp0+fHjhw4K+//vrss8926NABHZILFix4++230Yhkr++I+1aVXdu4ceMDDzyQnJyM5S3B5nK8xD2MR0afPn1uvPFGlHc2btw4d+5cXdexwDK+EbaZczqdP/74Y8eOHb/++musBoNDobzuOW9yy7IqVqz46KOP4nGWmJg4duxYDJpUZ2ghpwSlB6CONWHCBDQiezyeIUOGoClG1dZXBbHRvID2opEjR86bNw8NFFn2c4fzI1ObNWvWrl07NApt27bt66+/1nXd5/PZP49KamJi4rBhwzwej6qvYM/DR5kGGyqg0VzX9UmTJh08eBDXQJ06dVq2bJldDOsVJAYsUYc+A1cQWGioadOmffv2/eijj1atWtWvXz9l21HNJHK5QewOHhVRGiJ7KRUN64OtXbu2a9euzZs3x6QELJF0DffFI1bI6VAAWxwbAIwdO/a6667DoomzZs367LPP0HSAuQgoiTscjl27dnXo0GH58uWYAzV9+vRixYphQMITTzwxZcoUNIyiioDNF/GLmqZ98cUXHTt2/PfffwEAS7FC/sUj4eESGxuL9hwUG5944glUzDGBE7WWLVu2DB8+vHv37hs2bMAalhgOpKzYF1TS8VQaMWJE48aNcWTmzJmDDa3sfuxCLhko05Bpmvfcc8/QoUPT09MdDseGDRsGDBhw+PBhLH6lmmhi6js2b7ntttveeecd7JRgby2ZpfyBjiVd15955hm0WuB62759O6YU+P1+XEJut/vs2bODBg3CRkkhEoMiM6XlYCjXkSNHsP42ktCwYcOwxmfh4WYcBK/X269fv+nTp0dHR3u9XizaUbJkyW+//fann376/PPPBw0aVKJECVTF7OXHc699qlYZ6FwpXrw4aldKHbEHGqn68OvWrcNI6yVLljz55JN22zLhv8IKyoiERnBcHBEREVOmTImPj8edNnz48ClTppw4cQKTWXRdP3LkyMyZM9u1a/fnn38CQLFixVwuF4alFilSBCMlhg4dOnTo0PXr12OnT4fD4XQ6z5w5s3bt2nvvvXfAgAFnzpxhjEVFRTVs2BDtLRg6Def33sobUFRv2rTppEmT0Gx6+PDhfv36rVq1CnuaOxyOzz777IYbbnjnnXewXBJj7I477kDhNweKCrGVqw4wr732WkREBOZtPP300x6PB7eT3Z5eOBtE26OT8UyfOHHigAEDfD6fy+X6+eef27dvP3v27CNHjuACwBoYu3fvnjZtWqNGjRYsWIBWprp16yYkJIR46e3DpdKphBDNmjV744030IRy6NChAQMGrFy50jRNp9OJDPTbb7/17dt34cKFmqZhD/rwVaHCmVRA1wsvvIBNL/x+f/Pmzfv371/YSj1jJA8a3/r06fPdd98lJCSghnrs2LE333zz7NmzmNVoDxJVsWr2rnM5C1KKFbDdZvXq1XEd/vDDDxCMP7bvl61bt27evBljRlBRmz9/Pmbv06F/4QPnGic9zn0+X+vWrb/++uv+/fufPHkyJSVl6NChU6ZMadKkicvlSk1N3bhx49atW3GTV6tWbc6cOfXr1/f5fO3atVuyZEmvXr2OHDmiadqUKVO++OKLG264AR1cfr9/375969atQ6cFZh3PmTNnxYoVyC75JdPZPZbDhw9PTk5+4YUXAODPP//s2LFjq1atqlatKqX87rvvTNOMiopKS0vz+XwTJky44447li1bdrHhoWhHuummm+66665p06ZFRET89ttvH3zwweOPPx7Sj+iqqEePp8n06dOdTueMGTMAYOfOnX379q1Tp07Dhg1R9D527Ni///67fft2PKT8fv/gwYPvvvvuO++8MzvdSEn6CNM0H3vssdTU1LFjxwLAX3/91bFjx7Zt21apUoVznpiYuHz58pSUFACoVKnSI4888vjjj+egdanugZ999hmegxEREa+++ip2FkNjZqGyIEGwGcnNN988b968u+666/Dhww6HY9myZR06dPjyyy8bNGiAigLa9/O8C3DfxcXFtW/ffs2aNREREStWrFi3bl3z5s1VD1R0Whw6dAi1Fnt9382bN9erV6/wp4UXim1zDQNDKlHeX79+fc2aNXMYivbt2+/btw+9x+q/+/fv79OnT861qTnnnTt3/vvvv/HgxkUcHx+/ceNGFQVvfyQ84k+dOtWwYUO8QqdOndBkYf+kPf8As/CwLPMHH3yA9YdDngF/cLvd//vf/6SUf/31V0REBP7+0UcfxcMrh7HC51SJcv/++2/FihUx5DwyMvKPP/7AK6gexSq0xg4c6s8//1yJwzNnzlSxv+HvJaUcMGAAPnnFihX37NmT3XPilb/99ltV8GPKlCnqscPHTb0Uhs+bpjljxgyMIs0BxYoVe++99yzL+ueff4oVK4Y3wvRj5U+yvwLaoNRofPbZZ9jzNUvUqVPn77//Xr9+PQAgIb399tv4auri+JoHDx7ER8WPvfjii1LKtLS0LMc837cMPsOsWbPUJH766adZTiK+OK5Py7LS0tKklKtXr8aHx11TsWLF9evX44ayp/jkcGv7LH/wwQdq4eHXTdM8fPhw2bJlUZJr1KjRrl278Ap+vz89Pd3v93/11VfKvqccGHgp+1vgk5w9e7ZJkyY4Rx07dlQ7MXyupZRLlixRKvgbb7xxwW11NYL/F2gPvQjNmjVbv379a6+9duONN9qT4IsXL96pU6d33nnn+++/r1ixIkZ2YsSI3++vUKHC119/vWTJkr59+zZt2jRkz1evXr1r165ffPHFokWL6tSpYy/oqAwCly6OKU8vimZDhgz59ddfhw8fjnGKyp5TvXr1QYMGLV26dMSIEfgASmfPZQEyFRLu9XqrVav2zDPPqNSHRx555OzZs/ZgcCj0vQxxiasc5oEDB+7atWvcuHE333wzlqawk8ENN9zw9NNPr1+//uGHH0aNAc7POAvJfLbnIeM4+/3+u+66648//njqqafq1q2rfDmc8xo1ajz55JO4SFBpQLN4yApBT0ZaWtqgQYPQMZ6enn7zzTePGTMG+4IUtoh71K7UQY+lhW+88cZFixZVrVoVK2jt37+/R48eq1evxgHJm2sKFVO1EcqUKTNp0iS846ZNm1q1avXuu+/u2LEDDarYR0HVN1POIXsNV8J/14KkaqegaT4iIuKpp54aNmzYzp07Md3GNM0KFSrccMMNcH7BBlxJGM3GGGvdunXr1q0TExP379+/Z88evGxUVNR1112HBiWUZdD5XBCshucFOpCx9vKkSZNOnDjxzz//nDp1SkoZFRVVt27dChUqoJaNH1MKxEWFeeAdsZHDrFmzfvnlF5fLtX79+jFjxrz33nuqa8JVoYajBUaltum6/uKLL5qmuXfv3n///Tc5ORknumLFirVq1YqOjgYAr9frdDp1Xc/NVNrHAe9VuXLl1157bfjw4du3b09MTMTeebhOVIM/yCZwmTF2+vTp++67b/ny5VjOoXLlytOmTcO2POgCKWzDjnFxmHWMP/t8vgYNGixYsODOO+/cvHkzltXq3r377NmzO3TokLfwUHuFFdzOvXr1evrpp19++WXG2KlTpx577LGqVatWqlQJyxlgLyMl7wsh2rdv36JFC6C+OsQKEMz2UploPp8vKiqqcePG9orEWFoAlU1VXRmrp+E+RNMNVtpq1qxZiBUYbDHUdpt7voh1KtQSGUtV9pdSlihRokSJEuEPg8+s4v/yUIMMSVTX9ddff71169YYtTlt2rQWLVrce++96E7HFyzMFQLsjVpVqDt646tXr169evWQz6uCmhhpZm9LB+dXNwrJXFF+eJW/Urp06RBrlWphn3M4ZkxMTMuWLX/88UekgQ8++KBKlSpoK7efj4VnhNHfjsSAyx45rGbNmgsXLuzRo8emTZscDkdSUlLX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qU6ThISETz/9tG/fvr///vvBgwdvueWWV199tW7duvaacSqopnbt2tHR0QUn/Npz6bdt27Zs2TJ8RxX1pBxIub/mZbPs45g0b968Xbt2ixcv1jRtwYIFw4cPr1atGhrEVMsBNf44qhcM5YyPj7dLNtk1bmvZsmWnTp2+//57zvmiRYu2bdt23XXXqQ4HSk20XyQk5x8RIkXVqlWrVq1aikVQGktLSxs1alRycjKWG+jTp0+pUqXw1fDgDvHr5tfyMAzD7/c7HI79+/cvXLgQXSn25YEyYh4MqnAxnUsKIyvYh1txgPLG4DoI0fpDJAL8OSMjY9WqVfib0qVLYwUue6CRXbK23wtsdTIuA9Up+UWdUPjwY8eO3blzZ+hk6Hq7du1mzJhRsmRJVG9VvqXiwhAbkbKHhtzXHvmKFeft2Zu6rvfp06dq1ao5l53gnC9cuPDPP//M4WVxzB944IH33nsPcyw++uijHj16KOdeRkZGTEwM1odB805kZOQXX3yxbds2l8uFfrMsn2HVqlW4GJTxyi4+4+tUrFhx9uzZDz744LJly5KSkh588MHspua5554bP368fS3l7wJQF8QxwUKkYIswxne5/vrrb7311hyOezz+tm7d+s0336gNH24bLCAj0tChQ5EVDh069PXXX48dO1ZFy+C5phz7MTExlmW9+uqrGRkZWJFU7VD7hPr9/v3796sCZTkspBEjRixZsgQATp48+emnn7722mvoqsEVEhERoepiRUVFMcbefvvtM2fO2KV7eygKurX69OmDte1UlYi9e/eOGDFi6dKlWJSsVKlSTz75JAo3KIepTlkqWim/RlgVlkavPsYjKAkG57pBgwa33XZbiCQRchGsAzZv3jzVVyrf/QpQQFX9UEI8efJkjRo18EYlSpSwR+8JG1QcIWqv5cuXx7Fo2LAhet7tH0addOPGjVhPmDGG2wzl3xyqukop58+fr8bxgw8+wAJ2+fjW9oAzvKm9xYKK7cEaoijtxsXFFSlSJDY2Fo/XYcOGHT58WFnAMHAFj3W8CJb62rBhQ/HixXFgX3/99fDeGsgi//77b7Fixex5v7g9oqKiclm4FM9Zznm5cuV27NihrmwHDvu7774LAFh5H89f9YHExERsEcEYq1Spksfj+fbbb3NpRQGAUqVK/f333+FlPi3L8nq9+EidO3fmnEdGRkZE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FC5W28+pEOcnfh7nCzWh+Pj4F1988Y477sArx8fHt2vXDh+yevXqd999d5UqVdRZg9ZknN9wb7nqupNdJShkhWXLlm3fvh09h717954wYQIykK7rmCKuzF8hoVBoRsvfamhqDHHcqlev/sorrwwYMEBFwaF/SJ1B4TqQsicg0eLPl8eCpLpIWpbVsGFDlGSdTmcgEKhRowa2rAgEAq1atbrzzjuLFCliry+CZ7c9H1vFNaSmpip154IJAUKIWrVqjRgxYsKECTinlSpVatmyJd6rUaNGAwcORIur8tbg4leBsLfcckurVq2WLl2K5dF69uyJf23YsGGDBg3q1KmDCWtIbzjIKrhRuZdVoZqXX375yJEjUVFRqpdOniUGsJW6wXkvV67cK6+80rNnTyRdjFlHw5edq0JyF9QCUyLXVcMKaokod67D4Th79uw777zz2GOPYZaZsuXhQixfvnyDBg3Wrl2Lc/P2228PHDhQ6d1Kw0UZCgDi4uLwyAOAbdu2Ke3bvvKUUUIVdUE/pL15b0FEWOEG+/vvvydMmOB0Ov/5558PPvjgmWeeUSegvb+QPahGPWqvXr0++OADr9fLOZ88eXKPHj1atGiBBX/QGnPrrbeuWbMGkzOzdAzgC6oT/5133tm9e/e6deswX9oeiJVzZ1e7kBIittiN+0oqv+2221q1arVmzRrkudGjR9vT8XC+kObx81kqOiFdd7KMmkez1fLly9UrDB06FBdeeChkSHwwnjI+n2/evHmVKlUKsYldoq4AwcpXUsq+fftu3779xRdfVOFkygmEpu1wOrRLMCH9qXLuSJgvuo46swYPHjx79uxTp07hA8+YMcM+UyEHaA5dd+z7Nwe9R1kRpZT333//zJkzsasSALz//vtoRLVb2JR5wB5xjmv+2WefXbp0KZ4wL774IhZSDNHGVEytWh5o41WtDx0OxxdffPHxxx/HxcXllyiplrTyCN56663/93//9/jjjyP7qg1il3FDtmeI4b2AhIaCit3GEzwuLu6ZZ57BpWZZ1vDhw9999110B6nYABypcuXKvffee2+88cbAgQP3798/bNiw8FA2XGf4y2rVqmH3O4fDsWvXrsTERHvas134UgcB5/znn38uUOFLCXpCiI4dO/bp0web/3z++efHjx+39wrNriITLtb69es3btwY/e3Y/fWVV16pV69e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3r170fjWuHFjlZR7BYPVLlZTRGXu1VdfLV++PIYPvP7662vWrMGJzpcbYdpt2bJl58+f3717d1xdgUBg7ty5o0aN6tSpU8eOHQcNGvTxxx+fOHECv3L//fevWLECIxfRuGR/mELoAb6y84gZFbGxsRMmTMBd6fF4RowYcfr0aVVkooDujreOjIx8+eWXMbcrEAiMGTMG25BA9vUz7M+PpsUhQ4YsX778gQceUGUQsTcXEpv6/I033jh9+vQffvihWbNmqu12eJ/Uwr8BC69fIWQcMbcIXStRUVGzZs3q27dvWlqay+U6evRor1691q9fj6VysGlGLt0sqkRS/fr1v/766w0bNowZM6ZJkyalS5cuW7Zsy5YtX3755T/++GPy5MlutzskoekybCrlafD5fG3atHnzzTeFEC6Xa/bs2aNGjfJ4PC6XC6Ue/LzH4zEM49SpU5MmTVLxi61bt4bLWGr/0oFEqNwq6GPnnKenpw8dOvT48eP5FaqPnknLsooUKTJnzpz33nuvTJkyGPRslxvcbnelSpUGDx68YcOG6dOnY88Ae3beVcG1VwSqmGjnzp379++Ptt+tW7cOHz4cbOnuBXrrNm3aYNqB0+ncs2fPY489purc5LyRVaiF3++vW7futGnTNm3a9MEHH4wcObJ8+fJly5YtW7ZsqVKlypUr98QTT/z444/z58+///77cbfaE19Csoj/C0tFvzznozqO0dVsGMbkyZPPnDmzfPlyl8t1/PjxQYMGzZ49u3bt2n6/H/2HuTm7MVcLJ9IwjMaNGzdu3Pj//u//wo8Pv9+P3s7LSrnBegBIhw899NCpU6eee+45t9v9wQcf7N69++23377uuutQeNF1HV/86aef3r59e0REhMfjKVGiRMeOHcN7zhRmKGcgBKvgffnll8uXL3c4HP/888+wYcNU54B8uZfyTz788MMDBgxYu3btokWLsKIZ57xt27alS5du37492gpQAUcnsD2i126nvoJevkIFlGmULWjcuHFLly5NTEx0uVyzZs2qX7/+k08+WUBuZ/uthRBjxoz54Ycf9u/f73Q6v/nmmxdeeOGll166YGiQvcs39mquXLnykCFDAOCtt97K8isY3Y6quXK55xwCTrrCpXrekL1R8U9ISPjqq6/atm3r9XpjY2O3bdt2xx13bN26FUN0VHUwPD6y6x2m/HXY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K4pZAqJSfOjQIfQyhW7o9XrXrl07fvz4rVu3ko+Xc96pU6eqVavCieT1et1u9zXXXEOePU3TZs+e/corrzz//PNiPYZ9+/Y999xz2dnZIPGc8/Lly5ctW1Z0rjZv3rxr166LFi0i1Xj58uVt2rR5+umnb7zxRk3TMjMzx44dO2XKFAgOINwOh+Ouu+4iyrtv3769e/dSo5XatWt///33oImhDk+fzzdkyJBNmzZhNHSX7Nat2xWEVxXTYQuMbNasWdu2bVG5DzR648aN/fv3f//991u0aEG1i3Nzczdu3Pjoo4+ePHmSyoNzzm+77bYqVaoUJu+LQbogu02aNBk8ePAnn3wiehSKn9UMTpCcnExs0jTNL774omvXrtDKRT8/irx+8MEHn332GU7H6/UuXbq0b9++YWcrZnSDdnTu3Ll3794//fQTNQkvjo3OsqyDBw9mZmZShyXDMPbs2fPtt9/OmzdPDI2tUqUKaMEVcwVyIEMbq1u37osvvvjMM8/AaK5p2tGjRz/88MP33nuPkoSCjHuMsf3791MBqFD+R8UoS5YsiV7onTt3XrJkCe32+PHjq1at+sADD8THx1Mgw/nz51966aW1a9cinAEHjVg7Ur9ATMGMs7KyDh06FFpmMUjTRaAz+pCfOHHit99+g4Th8/kqVar03XffVa5cORSX8Ir7779/xYoVuNEXL15csWLFoEGDIphnqRIMEt179ux5++23f/vtt+jE5XA4Vq5c+f33399zzz2II8Ki7rnnnp9++ikrKws/9Hg8b7/99rp16x588EFUNNi+ffvYsWNXrVqFmcB92LRp006dOlEy0J+dx1oo+v7xrjuwMFDXncsN6J4xYwaG2r17d7ly5YpzN0jDUFW1RIkSO3bsQAAvNUhZvXp1TEwM9bLAr1q0aPH6669/9dVXX3zxxSuvvII4AXwLFHz00UdhEcKiIGOuX78+OjoaUg/13gn1T8InAWnlmWeegcACT9S4ceOQl4BvH3zwwchb+tJLLzHG4I2n56F4iho97CHr16/HY5jYbbfdFtqDl5pvoOsORS7eeeedYg7UunXrYmJiSLfDbJ1OZ/fu3T/77LMvvvji008/vf7668kyg91AbZyjR49SwS903WnZsiWpIygyQfIaYP/+/aVKlRILhiOhaebMmWG77tC68NXx48dLly5NMiYCBCPsKqyI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M0B2cTgcaK+G1dlsttTU1OHDh0Nogwrr8/ncbvfXX389duxYJFrjnlJkBMIgExMT33zzza+//prygcTCzOSmFZcP8yM56oM6oIitga6gyJh+FT0TqqrGxsYiLrM4weaQasXOw4FAICEhIScnB16dULpgWVb16tWvu+665s2bd+/evVy5ctQsU9ROQJEty+rcufPy5ctff/31mTNnejyesIUKqlatOmLEiCFDhkBrEaMLKBsTxLFz586tWrVaunTp+vXr586de+TIEUQytGjRonXr1j179oQFUHSdGYaxbNkyt9sN+mKaZo8ePVhBsc+wbByH3a1btxkzZiCeyufzrV+/Ho0BgopNAkFjY2MhQQcCgZiYmAjVQNH+CVTD6/UiFERsFQekfPnllzt27Pjhhx8uWLCATENBwSr9+/d/+umnqagydknM2IyOjna73UhYRR6DmJFP+H3LLbdMmDBh27ZtiJt0uVwkFhUWk4p6y5g/rkf79u1hag8qh0XRychqrFWrVqNGjQ4ePAgBDZ7Gm2++2ePxxMfH02xJYSWBg5SzO+6448cff9y/fz9ygKOiosS1mKaJVYeNFsP2RkVFdevWrXbt2n369IHNndzjEcqiieBwOFDtAG5zscQhVWNEcU2/31+9evXnnnvujTfeQH06xtjWrVtnzpwJ/5PT6USOd3FK/uEuUHcQ4oKVKlX65ZdfPv7446+//jotLU1EFSJJiYmJzz777KOPPgpMcDgcFDJgmmZCQgJ0zbCablhvMxK4OOfLli2LioqCZcbr9YLxh8pbIg3B/fruu+9Aanw+3759+06fPp2UlKQoCqIcYUQNzdUXM926dOly//33f//993a7HaktCxYsWL16dZs2bcBvoCIrivLYY4/16dNnxYoVS5YsWbJkCaIKDcPo1q1b69at+/Xrl5SUBJkYZZFEmoazhqSLGDA6awgBJHLBEuVyuSC5Yn+uIH7hj4Y9iUnkYth4ceLNqWxWVFQU8AP8nwKHQm0I2GiRi1DEGNklEbNEdxI3fM+ePb/99tv8+fNTU1PpJ82bN2/Xrl379u1R0FsMBRNbIzChzSRZ0iFHk2oCQgadmjQnTACF27BAv9+PgioUjhnkZCOtiNI1SMSmfh0UQE2ZSpD1sChYycJaYChqnsKlSK4UbS8QKrHJhw4dWrFixa+//pqeno5c38qVK99w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" alt="Institute of Professional Education and Research Bhopal logo">
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
                        full_name = st.text_input("Full Name", placeholder="Enter your full name")
                        scholar_id = st.text_input("Scholar ID")
                        email = st.text_input("Email ID", placeholder="yourname@iper.ac.in")
                    with col2:
                        create_password = st.text_input("Create Password", type="password")
                        confirm_password = st.text_input("Confirm Password", type="password")

                    signup_submitted = st.form_submit_button("Create Student Account", use_container_width=True)

                if signup_submitted:
                    normalized_email = email.strip().lower()
                    if not full_name.strip():
                        st.error("Please enter your Full Name.")
                    elif len(full_name.strip().split()) < 2:
                        st.error("Please enter your complete name, including first and last name.")
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
                            full_name, scholar_id, normalized_email, create_password
                        )
                        if ok:
                            st.success(message)
                            st.info("Please open the Student Sign In tab and log in with your @iper.ac.in email ID.")
                        else:
                            st.error(message)

            st.markdown('<div class="auth-footer">IPER Student Placement & Career Readiness Portal</div></div>', unsafe_allow_html=True)


init_database()
migrate_local_sqlite_to_postgres()

if not DATABASE_URL:
    require_persistent_database_for_production()

FFMPEG_PATH = shutil.which("ffmpeg")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    render_authentication_panel()
    st.stop()

# Keep the logged-in student's full name as the permanent personalized display name.
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
st.sidebar.markdown(f"### Welcome, {st.session_state.get('first_name', 'Student')}")
st.sidebar.caption(f"Scholar ID: {st.session_state.get('scholar_id', 'N/A')}")
st.sidebar.markdown("Career Readiness & Interview Hub")
if not FFMPEG_PATH:
    st.sidebar.warning("FFmpeg is not installed. Audio/video transcription will not work until FFmpeg is added to the deployment environment.")
st.sidebar.markdown("---")

selected_nav = st.sidebar.radio(
    "MAIN MENU",
    [
        "Progress",
        "Industry & Company Insights",
        "About Myself",
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

# SECTION 0: PROGRESS
if selected_nav == "Progress":
    st.title("Progress")
    st.caption("A quick view of your placement-practice journey, activity and next steps.")

    scholar_id = st.session_state.get("scholar_id", "")
    student_id = st.session_state.get("student_id")
    history = st.session_state.get("history", [])

    # Refresh persisted interview attempts so progress survives login/session refreshes.
    if student_id:
        try:
            persisted_attempts = load_student_attempts(student_id)
            if persisted_attempts:
                history = persisted_attempts
                st.session_state["history"] = persisted_attempts
        except Exception:
            pass

    pi_attempts = len(history)
    pi_scores = [float(item.get("Score", 0) or 0) for item in history]
    pi_minutes = sum(float(item.get("DurationSeconds", 0) or 0) for item in history) / 60.0

    gd_attempts = 0
    gd_scores = []
    gd_minutes = 0.0
    if scholar_id:
        try:
            gd_assessments = get_gd_video_assessments(scholar_id)
            gd_attempts = len(gd_assessments)
            for assessment in gd_assessments:
                report = assessment.get("report_json") or {}
                participant_report = next(
                    (p for p in (report.get("Participants") or [])
                     if str(p.get("Name", "")).strip().lower() == str(st.session_state.get("candidate_name", "")).strip().lower()),
                    None
                )
                if participant_report and participant_report.get("OverallScore") is not None:
                    gd_scores.append(float(participant_report.get("OverallScore", 0) or 0) * 10 if float(participant_report.get("OverallScore", 0) or 0) <= 10 else float(participant_report.get("OverallScore", 0) or 0))
                gd_minutes += float(assessment.get("duration_seconds", 0) or 0) / 60.0
        except Exception:
            pass

    total_minutes = pi_minutes + gd_minutes
    all_scores = pi_scores + gd_scores
    average_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

    # Progress is intentionally activity-based: 20 completed practice assessments is the first milestone.
    target_attempts = 20
    progress_pct = min(100, round(((pi_attempts + gd_attempts) / target_attempts) * 100))

    st.markdown("### Placement Readiness Progress")
    st.progress(progress_pct, text=f"{progress_pct}% of your first {target_attempts} practice assessments completed")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Minutes Utilized", f"{total_minutes:.1f} min")
    m2.metric("PI Attempted", pi_attempts)
    m3.metric("GD Attempted", gd_attempts)
    m4.metric("Average Score", f"{average_score:.1f} / 100" if all_scores else "—")

    st.markdown("### Your Way Ahead")
    suggestions = []
    if pi_attempts == 0:
        suggestions.append("Complete your first PI practice session and establish a baseline score.")
    elif average_score < 60:
        suggestions.append("Focus first on answer structure, clarity and confidence before increasing practice volume.")
    elif average_score < 75:
        suggestions.append("Continue regular PI practice and strengthen examples from your resume, internship and projects.")
    else:
        suggestions.append("Maintain your current performance and practise company-specific and advanced interview questions.")
    if gd_attempts == 0:
        suggestions.append("Attempt at least one GD and practise concise points, active listening and balanced participation.")
    elif gd_attempts < 3:
        suggestions.append("Complete a few more GDs so your participation and communication patterns become consistent.")
    else:
        suggestions.append("Review your GD feedback after each attempt and target one specific improvement in the next discussion.")
    if total_minutes < 30:
        suggestions.append("Aim for at least 30 minutes of meaningful placement practice before your next review.")
    else:
        suggestions.append("Keep a steady weekly practice routine rather than relying on last-minute preparation.")

    for suggestion in suggestions[:3]:
        st.markdown(f"• {suggestion}")

    st.markdown("### Activity Snapshot")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("PI Practice Minutes", f"{pi_minutes:.1f}")
    with c2:
        st.metric("GD Assessment Minutes", f"{gd_minutes:.1f}")

    if history:
        st.markdown("### PI Score Trend")
        trend = pd.DataFrame({"Attempt": range(1, len(pi_scores) + 1), "Score": pi_scores})
        st.line_chart(trend.set_index("Attempt"))
    else:
        st.info("Your progress will start building as soon as you complete your first PI or GD practice assessment.")

# SECTION 1: INDUSTRY & COMPANY INSIGHTS
elif selected_nav == "Industry & Company Insights":
    st.title("Industry & Company Insights")
    st.caption("Explore sectors in India and the companies documented in IPER's 2025–26 placement ecosystem. Use the information here to prepare for roles, interviews and campus recruitment.")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#433B86,#1E3A8A);padding:26px 30px;border-radius:14px;color:white;margin:10px 0 22px 0;">
      <div style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;opacity:.85;font-weight:700;">IPER Career Intelligence</div>
      <div style="font-size:30px;font-weight:800;margin-top:5px;">Know the Sector. Know the Company. Know the Role.</div>
      <div style="font-size:15px;margin-top:8px;opacity:.94;max-width:900px;">Understand India's business sectors, explore companies appearing in IPER's published 2025–26 recruiter list, and identify the knowledge and skills to prepare before a placement drive.</div>
    </div>
    """, unsafe_allow_html=True)

    # IBEF reference pages used for sector-level learning. The portal contains
    # original student-oriented summaries; it does not reproduce IBEF articles.
    IBEF_SECTOR_URLS = {
        "Banking & Financial Services": "https://www.ibef.org/industry/banking-india",
        "Insurance": "https://www.ibef.org/industry/insurance-presentation",
        "Financial Services": "https://www.ibef.org/industry/financial-services-presentation",
        "FMCG & Food": "https://www.ibef.org/industry/fmcg-presentation",
        "Paints & Building Materials": "https://www.ibef.org/industry/paints-india",
        "Consumer Durables & Electricals": "https://www.ibef.org/industry/consumer-durables-presentation",
        "Technology & IT Services": "https://www.ibef.org/industry/information-technology-india",
        "EdTech & Education": "https://www.ibef.org/industry/education-presentation",
        "Telecom & Digital Services": "https://www.ibef.org/industry/telecommunications",
        "Automotive & Mobility": "https://www.ibef.org/industry/automobiles-presentation",
        "Manufacturing & Engineering": "https://www.ibef.org/industry/manufacturing-sector-india",
        "Textiles": "https://www.ibef.org/industry/textiles",
        "Hospitality & Tourism": "https://www.ibef.org/industry/tourism-hospitality-india",
        "Media & Entertainment": "https://www.ibef.org/industry/media-entertainment-india",
        "Retail & E-commerce": "https://www.ibef.org/industry/retail-india",
        "Real Estate & Housing Finance": "https://www.ibef.org/industry/real-estate-india",
        "Renewable Energy": "https://www.ibef.org/industry/renewable-energy",
        "HR & Recruitment Services": "https://www.ibef.org/industry/services",
        "Business & Professional Services": "https://www.ibef.org/industry/services",
        "Diversified / Conglomerate": "https://www.ibef.org/index.php/industry.aspx",
    }

    SECTOR_INFO = {
        "Banking & Financial Services": {
            "about": "Banks and financial institutions provide deposits, lending, payments, investment and related financial services. MBA roles commonly span relationship management, sales, credit, operations, analytics and HR.",
            "roles": "Relationship Management • Credit & Risk • Sales • Financial Analysis • Operations • HR",
            "skills": "Financial literacy • Excel • Communication • Customer handling • Analytical thinking • Sales orientation",
            "prepare": "RBI basics, banking products, KYC, credit, digital banking, financial inclusion and basic financial statements."
        },
        "Insurance": {
            "about": "Insurance businesses manage risk through life and general insurance products, distribution networks, customer servicing and claims-related processes.",
            "roles": "Agency/Channel Sales • Relationship Management • Underwriting Support • Operations • Customer Service • HR",
            "skills": "Communication • Financial literacy • Relationship management • Compliance awareness • Data interpretation",
            "prepare": "Life-insurance concepts, risk pooling, premiums, policy servicing, distribution, bancassurance and customer needs."
        },
        "Financial Services": {
            "about": "Financial-services companies support investing, securities, wealth, lending and other capital-market activities.",
            "roles": "Wealth Management • Broking • Research Support • Sales • Operations • Client Servicing",
            "skills": "Financial markets • Excel • Analytical thinking • Communication • Client management",
            "prepare": "Equity/debt markets, mutual funds, risk-return, basic valuation, investor behaviour and financial regulation."
        },
        "FMCG & Food": {
            "about": "FMCG and food companies compete through brands, distribution, pricing, product quality and consumer understanding, creating strong opportunities in sales and marketing.",
            "roles": "Brand Management • Sales • Trade Marketing • Distribution • Category Management • HR",
            "skills": "Consumer insight • Market research • Negotiation • Channel management • Data interpretation",
            "prepare": "4Ps/7Ps, consumer behaviour, brand positioning, distribution, rural markets, category management and e-commerce."
        },
        "Paints & Building Materials": {
            "about": "This sector connects manufacturing with dealer networks, construction demand, distribution, product innovation and consumer-facing brands.",
            "roles": "Sales • Marketing • Dealer Management • Supply Chain • Finance • HR",
            "skills": "Channel sales • Territory planning • Relationship management • Negotiation • Market analysis",
            "prepare": "Dealer/distributor economics, B2B/B2C selling, construction demand, territory management and product positioning."
        },
        "Consumer Durables & Electricals": {
            "about": "Consumer-durable and electrical businesses combine product development, manufacturing, distribution, retail and customer service.",
            "roles": "Product Marketing • Sales • Distribution • Supply Chain • Category Management • Finance",
            "skills": "Product knowledge • Channel management • Consumer insight • Analytics • Negotiation",
            "prepare": "Product positioning, distribution, channel margins, consumer demand, retail and after-sales service."
        },
        "Technology & IT Services": {
            "about": "India's technology sector includes IT services, software, analytics, digital platforms and technology-enabled business processes.",
            "roles": "Business Development • Product • Analytics • Customer Success • Consulting Support • HR",
            "skills": "Digital fluency • Data literacy • Problem solving • Presentation • Adaptability",
            "prepare": "AI, cloud, analytics, cybersecurity, SaaS, digital transformation and technology-driven business models."
        },
        "EdTech & Education": {
            "about": "Education and EdTech organisations deliver learning through institutions, digital platforms, content, counselling and technology-enabled services.",
            "roles": "Business Development • Academic Operations • Sales • Customer Success • Marketing • HR",
            "skills": "Communication • Consultative selling • Presentation • Customer handling • Digital literacy",
            "prepare": "Education business models, learner journeys, B2C/B2B sales, digital learning, customer acquisition and retention."
        },
        "Telecom & Digital Services": {
            "about": "Telecom and digital-service businesses combine connectivity, customer acquisition, network-led services and digital products.",
            "roles": "Sales • Product • Digital Marketing • Customer Experience • Business Analytics • HR",
            "skills": "Digital literacy • Customer analytics • Communication • Product thinking • Data interpretation",
            "prepare": "5G, ARPU, customer acquisition/retention, digital services, network economics and competitive dynamics."
        },
        "Automotive & Mobility": {
            "about": "Automotive and mobility businesses span vehicles, components, tyres, dealerships, finance, after-sales and increasingly EV ecosystems.",
            "roles": "Sales • Marketing • Dealer Management • Operations • Supply Chain • Finance",
            "skills": "Process thinking • Supply chain basics • Cost awareness • Customer orientation • Data analysis",
            "prepare": "EV transition, auto supply chains, dealer economics, mobility trends, inventory, quality and customer experience."
        },
        "Manufacturing & Engineering": {
            "about": "Manufacturing and engineering companies convert materials and technology into industrial, consumer or infrastructure products and services.",
            "roles": "Operations • Supply Chain • Procurement • Sales • Project Management • Finance",
            "skills": "Process thinking • Costing • Quality • Supply chain • Excel/data analysis • Problem solving",
            "prepare": "Capacity, quality, inventory, procurement, Industry 4.0, automation, productivity and cost management."
        },
        "Textiles": {
            "about": "Textiles covers fibre, yarn, fabric, apparel and related value chains, with management opportunities across manufacturing, merchandising, sourcing and sales.",
            "roles": "Merchandising • Sales • Operations • Supply Chain • Procurement • HR",
            "skills": "Cost awareness • Supply chain • Negotiation • Quality • Market understanding",
            "prepare": "Textile value chains, exports, sourcing, production planning, sustainability and fashion/consumer demand."
        },
        "Hospitality & Tourism": {
            "about": "Hospitality businesses combine accommodation, food service, guest experience, sales and revenue management.",
            "roles": "Sales • Revenue Management • Guest Experience • Operations • Marketing • HR",
            "skills": "Communication • Service orientation • Problem solving • Sales • Customer experience",
            "prepare": "Service quality, occupancy, yield/revenue management, customer experience, digital bookings and tourism demand."
        },
        "Media & Entertainment": {
            "about": "Media businesses operate across publishing, news, advertising, digital content, audience engagement and entertainment.",
            "roles": "Media Sales • Marketing • Advertising • Business Development • Content Operations • HR",
            "skills": "Storytelling • Sales • Presentation • Digital marketing • Audience understanding",
            "prepare": "Advertising models, audience metrics, digital media, content monetisation and changing consumer attention."
        },
        "Retail & E-commerce": {
            "about": "Retail combines merchandising, stores, customer experience, omnichannel commerce, pricing and supply-chain execution.",
            "roles": "Retail Operations • Sales • Category Management • Marketing • Customer Experience • Supply Chain",
            "skills": "Customer orientation • Sales • Merchandising • Retail analytics • Inventory management",
            "prepare": "Omnichannel retail, customer experience, merchandising, inventory, e-commerce, quick commerce and store economics."
        },
        "Real Estate & Housing Finance": {
            "about": "Real-estate and housing-finance businesses connect property development, home finance, customer acquisition, sales and project economics.",
            "roles": "Sales • Relationship Management • Project Support • Credit • Marketing • Customer Service",
            "skills": "Financial literacy • Sales • Negotiation • Customer handling • Market analysis",
            "prepare": "Home loans, property markets, credit basics, customer acquisition, project economics and regulatory awareness."
        },
        "Renewable Energy": {
            "about": "Renewable-energy businesses develop and deploy solar and other clean-energy solutions, creating opportunities across sales, project management and operations.",
            "roles": "Business Development • Project Management • Sales • Operations • Procurement • Finance",
            "skills": "Commercial awareness • Project thinking • Data analysis • Negotiation • Sustainability literacy",
            "prepare": "Solar economics, project lifecycle, energy transition, procurement, financing and sustainability."
        },
        "HR & Recruitment Services": {
            "about": "Recruitment and staffing businesses connect employers with talent and manage sourcing, screening, staffing and workforce solutions.",
            "roles": "Recruitment • Business Development • Client Servicing • Operations • Talent Acquisition",
            "skills": "Communication • Sourcing • Interviewing • CRM • Relationship management",
            "prepare": "Talent acquisition, staffing models, recruitment metrics, client management and candidate experience."
        },
        "Business & Professional Services": {
            "about": "Professional-services firms provide specialised business, consulting, technology, staffing or operational services to organisations.",
            "roles": "Consulting Support • Business Development • Client Servicing • Operations • Analytics • HR",
            "skills": "Problem solving • Communication • Excel • Presentation • Client management",
            "prepare": "Understand the firm's service model, target clients, value proposition, delivery process and role-specific KPIs."
        },
        "Diversified / Conglomerate": {
            "about": "Diversified groups operate across multiple business lines, giving MBA students exposure to different functions, markets and operating models.",
            "roles": "Marketing • Finance • HR • Operations • Business Development • Corporate Functions",
            "skills": "Business awareness • Analytical thinking • Communication • Adaptability • Cross-functional understanding",
            "prepare": "Research the specific business division, role, competitors, customers and current business priorities mentioned in the placement JD."
        },
    }

    # Complete recruiter list shown in IPER's 2025–26 placement graphic.
    # Source: IPER Placements 2026 page, image "Cos-at-IPER-2025-26.png".
    IPER_2026_COMPANIES = [
        (1, "Axis Bank Ltd.", "Banking & Financial Services"),
        (2, "Bajaj Life Insurance Ltd.", "Insurance"),
        (3, "DCB Bank Ltd.", "Banking & Financial Services"),
        (4, "HDFC Life Insurance Co. Ltd.", "Insurance"),
        (5, "HDFC Life Insurance Co. Ltd. (HR)", "Insurance"),
        (6, "ICICI Prudential Life Insurance Co. Ltd. (Gujarat)", "Insurance"),
        (7, "ICICI Prudential Life Insurance Co. Ltd. (MP)", "Insurance"),
        (8, "ICICI Prudential Life Insurance Co. Ltd. (Rajasthan)", "Insurance"),
        (9, "Teleperformance India Pvt. Ltd.", "Business & Professional Services"),
        (10, "Asian Paints", "Paints & Building Materials"),
        (11, "Berger Paints India Ltd.", "Paints & Building Materials"),
        (12, "Ceasefire Industries Pvt. Ltd.", "Manufacturing & Engineering"),
        (13, "Home First Finance Co. India Ltd.", "Real Estate & Housing Finance"),
        (14, "KMV Ventures Pvt. Ltd.", "Business & Professional Services"),
        (15, "Havells India Ltd.", "Consumer Durables & Electricals"),
        (16, "Indigo Paints Ltd.", "Paints & Building Materials"),
        (17, "Methodex Systems Pvt. Ltd.", "Manufacturing & Engineering"),
        (18, "The-H Digital Solutions Pvt. Ltd.", "Technology & IT Services"),
        (19, "IndiaMART InterMESH Ltd.", "Technology & IT Services"),
        (20, "CarWale (CarTrade Tech Ltd.)", "Technology & IT Services"),
        (21, "Yash Technologies Pvt. Ltd.", "Technology & IT Services"),
        (22, "AISECT Ltd.", "EdTech & Education"),
        (23, "Bhanzu", "EdTech & Education"),
        (24, "Edukyu Pvt. Ltd.", "EdTech & Education"),
        (25, "Trounsoler Ed-Tech Services Pvt. Ltd.", "EdTech & Education"),
        (26, "Jaro Education", "EdTech & Education"),
        (27, "Learning Shala", "EdTech & Education"),
        (28, "PlanetSpark", "EdTech & Education"),
        (29, "Bajaj Life Insurance Ltd. (additional source entry)", "Insurance"),
        (30, "PREPOCA (Limeam Eduserver)", "EdTech & Education"),
        (31, "Step UP Academy", "EdTech & Education"),
        (32, "Sygnific Careers Pvt. Ltd.", "HR & Recruitment Services"),
        (33, "Tata ClassEdge Ltd.", "EdTech & Education"),
        (34, "Toprankers Edtech Solutions Pvt. Ltd.", "EdTech & Education"),
        (35, "Info India Ltd. (Naukri.com)", "Technology & IT Services"),
        (36, "Eastman Auto", "Automotive & Mobility"),
        (37, "XL Dynamics India Pvt. Ltd.", "Financial Services"),
        (38, "Gujarat Cooperative Milk Marketing Federation Ltd. (Amul)", "FMCG & Food"),
        (39, "Haleon Plc", "FMCG & Food"),
        (40, "Himalaya Wellness Co.", "FMCG & Food"),
        (41, "Majestic Basmati Rice Pvt. Ltd.", "FMCG & Food"),
        (42, "Mahindra Holidays & Resorts India Ltd.", "Hospitality & Tourism"),
        (43, "Marriott International India", "Hospitality & Tourism"),
        (44, "Artech Infosystems Pvt. Ltd.", "HR & Recruitment Services"),
        (45, "Collabera Services Pvt. Ltd.", "HR & Recruitment Services"),
        (46, "Futur Staffing Solutions Pvt. Ltd.", "HR & Recruitment Services"),
        (47, "Sarthee Consultancy", "HR & Recruitment Services"),
        (48, "American Chase", "HR & Recruitment Services"),
        (49, "Anaxee Digital Runners Pvt. Ltd.", "Business & Professional Services"),
        (50, "Cogent Infotech", "Technology & IT Services"),
        (51, "Netlink Software Pvt. Ltd.", "Technology & IT Services"),
        (52, "Tata Consultancy Services Ltd.", "Technology & IT Services"),
        (53, "Yash Technologies Pvt. Ltd. (additional source entry)", "Technology & IT Services"),
        (54, "Toprankers Edtech Solutions Pvt. Ltd. (additional source entry)", "EdTech & Education"),
        (55, "KMV Ventures Pvt. Ltd. (additional source entry)", "Business & Professional Services"),
        (56, "Bajaj Life Insurance Ltd. (additional source entry)", "Insurance"),
        (57, "Bhaskar Industries Pvt. Ltd.", "Manufacturing & Engineering"),
        (58, "Impression Furniture Industries Pvt. Ltd.", "Manufacturing & Engineering"),
        (59, "Impression Furniture Industries Pvt. Ltd. (additional source entry)", "Manufacturing & Engineering"),
        (60, "Motilal Oswal", "Financial Services"),
        (61, "MPM Ltd.", "Manufacturing & Engineering"),
        (62, "Shakesteller Energy Solutions Pvt. Ltd.", "Renewable Energy"),
        (63, "Trident Group", "Textiles"),
        (64, "DB Corp Ltd. (Dainik Bhaskar)", "Media & Entertainment"),
        (65, "The Times Group (Delhi/Mumbai)", "Media & Entertainment"),
        (66, "The Times Group (Vadodara)", "Media & Entertainment"),
        (67, "Aditya Capital Pvt. Ltd.", "Financial Services"),
        (68, "Bajaj Finserv Ltd.", "Financial Services"),
        (69, "Bajaj Housing Finance Ltd.", "Real Estate & Housing Finance"),
        (70, "Home First Finance Co. India Ltd. (additional source entry)", "Real Estate & Housing Finance"),
        (71, "PlanetSpark (additional source entry)", "EdTech & Education"),
        (72, "ICICI Securities Ltd.", "Financial Services"),
        (73, "India Shelter Finance Corporation Ltd.", "Real Estate & Housing Finance"),
        (74, "Motilal Oswal (additional source entry)", "Financial Services"),
        (75, "Motilal Oswal (source entry)", "Financial Services"),
        (76, "NJ India Invest Pvt. Ltd.", "Financial Services"),
        (77, "Ashiana Housing Ltd.", "Real Estate & Housing Finance"),
        (78, "Ashiana Housing Ltd. (additional source entry)", "Real Estate & Housing Finance"),
        (79, "CarWale (CarTrade Tech Ltd.) (additional source entry)", "Technology & IT Services"),
        (80, "MoneyOne Consulting Pvt. Ltd.", "Financial Services"),
        (81, "SCG Group", "Business & Professional Services"),
        (82, "Deloitte Consulting India Pvt. Ltd.", "Business & Professional Services"),
        (83, "Aditya Birla Lifestyle Brands Ltd.", "Retail & E-commerce"),
        (84, "Avenue Supermarts (DMart)", "Retail & E-commerce"),
        (85, "Avenue Supermarts Ltd. (DMart) (additional source entry)", "Retail & E-commerce"),
        (86, "Bluestone Jewellery & Lifestyle Ltd.", "Retail & E-commerce"),
        (87, "Pantaloons Fashion & Retail Ltd.", "Retail & E-commerce"),
        (88, "Parnalan Fashion & Retail Ltd. (Calvin Klein & Tommy Hilfiger)", "Retail & E-commerce"),
        (89, "SolarSquare Energy Pvt. Ltd.", "Renewable Energy"),
        (90, "SolarSquare Energy Pvt. Ltd. (additional source entry)", "Renewable Energy"),
    ]

    # Clean display names for repeated source entries while preserving the fact
    # that IPER's published graphic contains repeated entries/locations.
    def display_company_name(name):
        return re.sub(r"\s*\(additional source entry\)$|\s*\(source entry\)$", "", name).strip()

    # Company-level student preparation notes. These are intentionally concise;
    # sector-level information is referenced to IBEF, while recruiter presence is
    # sourced from IPER's published placement material.
    COMPANY_NOTES = {
        "Axis Bank Ltd.": "Private-sector banking company; prepare banking products, relationship management, credit basics, sales and digital banking.",
        "DCB Bank Ltd.": "Banking and financial-services employer; prepare retail/corporate banking basics, customer acquisition, credit and operations.",
        "HDFC Life Insurance Co. Ltd.": "Life-insurance business; prepare insurance products, distribution, relationship management, customer servicing and sales.",
        "ICICI Prudential Life Insurance Co. Ltd.": "Life-insurance business; focus on insurance concepts, financial planning, customer needs and distribution channels.",
        "Asian Paints": "Paints and coatings business; focus on brand management, dealer networks, territory sales, consumer insight and distribution.",
        "Berger Paints India Ltd.": "Paints and coatings business; prepare channel sales, dealer management, marketing, supply chain and consumer demand.",
        "Havells India Ltd.": "Electrical and consumer-durable business; prepare product marketing, distribution, retail, channel management and customer experience.",
        "Indigo Paints Ltd.": "Paints and coatings business; focus on branding, dealer/channel management, sales and market expansion.",
        "IndiaMART InterMESH Ltd.": "B2B digital marketplace; prepare digital business models, B2B sales, lead generation, customer acquisition and retention.",
        "CarWale (CarTrade Tech Ltd.)": "Digital automotive marketplace; prepare digital marketing, marketplace economics, customer acquisition and automotive retail trends.",
        "Yash Technologies Pvt. Ltd.": "Technology and digital-services employer; prepare consulting/service delivery, business development, analytics and digital transformation.",
        "Tata Consultancy Services Ltd.": "Large IT-services and consulting organisation; prepare digital transformation, client management, analytics, consulting and business processes.",
        "Netlink Software Pvt. Ltd.": "Technology/software-services employer; prepare client delivery, technology-enabled services, analytics and business development.",
        "Deloitte Consulting India Pvt. Ltd.": "Professional-services and consulting employer; prepare structured problem solving, case thinking, data interpretation and presentation.",
        "Gujarat Cooperative Milk Marketing Federation Ltd. (Amul)": "FMCG/dairy cooperative and consumer brand; prepare brand management, distribution, rural markets, sales and consumer behaviour.",
        "Haleon Plc": "Consumer-health business; prepare brand management, consumer insight, sales, category thinking and healthcare-adjacent consumer markets.",
        "Himalaya Wellness Co.": "Consumer wellness and personal-care business; prepare brand positioning, consumer behaviour, distribution and digital marketing.",
        "Mahindra Holidays & Resorts India Ltd.": "Hospitality/leisure business; prepare customer experience, membership sales, service quality and revenue management.",
        "Marriott International India": "Hospitality employer; prepare guest experience, service operations, sales, revenue management and hotel business fundamentals.",
        "Trident Group": "Diversified textile/home-textile business; prepare manufacturing, exports, supply chain, merchandising, sales and operations.",
        "DB Corp Ltd. (Dainik Bhaskar)": "Media organisation; prepare advertising sales, audience engagement, digital media, content monetisation and business development.",
        "The Times Group": "Media and communications group; prepare media sales, advertising, digital content, audience metrics and business development.",
        "Bajaj Finserv Ltd.": "Diversified financial-services group; prepare lending, insurance, investment, digital finance and customer acquisition.",
        "Bajaj Housing Finance Ltd.": "Housing-finance business; prepare home loans, credit, customer acquisition, relationship management and financial analysis.",
        "ICICI Securities Ltd.": "Securities and investment-services business; prepare capital markets, client servicing, wealth/investment products and financial literacy.",
        "India Shelter Finance Corporation Ltd.": "Housing-finance business; prepare credit, affordable housing, customer acquisition, field sales and financial inclusion.",
        "NJ India Invest Pvt. Ltd.": "Investment and financial-services business; prepare mutual funds/investment concepts, client servicing, sales and financial planning.",
        "Ashiana Housing Ltd.": "Real-estate developer; prepare real-estate sales, customer experience, project economics, marketing and relationship management.",
        "Aditya Birla Lifestyle Brands Ltd.": "Lifestyle and fashion retail business; prepare brand management, merchandising, retail operations, customer experience and omnichannel retail.",
        "Avenue Supermarts (DMart)": "Large-format value retail business; prepare retail operations, merchandising, inventory, procurement, pricing and customer behaviour.",
        "Bluestone Jewellery & Lifestyle Ltd.": "Omnichannel jewellery retail business; prepare digital commerce, customer experience, merchandising, marketing and retail analytics.",
        "Pantaloons Fashion & Retail Ltd.": "Fashion retail business; prepare merchandising, category management, store operations, marketing and customer experience.",
        "SolarSquare Energy Pvt. Ltd.": "Solar-energy solutions business; prepare clean-energy markets, sales, project economics, customer acquisition and sustainability.",
    }

    # Official company / "About Us" websites for student company research.
    COMPANY_WEBSITES = {
        "Axis Bank Ltd.": "https://www.axisbank.com/", "Bajaj Life Insurance Ltd.": "https://www.bajajlifeinsurance.com/", "DCB Bank Ltd.": "https://www.dcbbank.com/",
        "HDFC Life Insurance Co. Ltd.": "https://www.hdfclife.com/", "ICICI Prudential Life Insurance Co. Ltd.": "https://www.iciciprulife.com/", "Teleperformance India Pvt. Ltd.": "https://www.teleperformance.com/",
        "Asian Paints": "https://www.asianpaints.com/", "Berger Paints India Ltd.": "https://www.bergerpaints.com/", "Ceasefire Industries Pvt. Ltd.": "https://www.ceasefire.in/",
        "Home First Finance Co. India Ltd.": "https://www.homefirstindia.com/", "KMV Ventures Pvt. Ltd.": "https://kmvventures.com/", "Havells India Ltd.": "https://www.havells.com/",
        "Indigo Paints Ltd.": "https://indigopaints.com/", "Methodex Systems Pvt. Ltd.": "https://www.methodexsystems.com/", "The-H Digital Solutions Pvt. Ltd.": "https://the-h.com/",
        "IndiaMART InterMESH Ltd.": "https://www.indiamart.com/", "CarWale (CarTrade Tech Ltd.)": "https://www.cartradetech.com/", "Yash Technologies Pvt. Ltd.": "https://www.yash.com/",
        "AISECT Ltd.": "https://aisect.org/", "Bhanzu": "https://bhanzu.com/", "Edukyu Pvt. Ltd.": "https://edukyu.com/", "Trounsoler Ed-Tech Services Pvt. Ltd.": "https://trounsoler.com/",
        "Jaro Education": "https://www.jaroeducation.com/", "Learning Shala": "https://learningshala.in/", "PlanetSpark": "https://www.planetspark.in/", "PREPOCA (Limeam Eduserver)": "https://prepoca.com/",
        "Step UP Academy": "https://stepupacademyindia.com/", "Sygnific Careers Pvt. Ltd.": "https://sygnificcareers.com/", "Tata ClassEdge Ltd.": "https://www.tataclassedge.com/", "Toprankers Edtech Solutions Pvt. Ltd.": "https://www.toprankers.com/",
        "Info India Ltd. (Naukri.com)": "https://www.naukri.com/", "Eastman Auto": "https://www.eastmanauto.com/", "XL Dynamics India Pvt. Ltd.": "https://www.xldynamics.com/",
        "Gujarat Cooperative Milk Marketing Federation Ltd. (Amul)": "https://www.amul.com/", "Haleon Plc": "https://www.haleon.com/", "Himalaya Wellness Co.": "https://himalayawellness.in/",
        "Majestic Basmati Rice Pvt. Ltd.": "https://www.majesticbasmati.com/", "Mahindra Holidays & Resorts India Ltd.": "https://www.clubmahindra.com/", "Marriott International India": "https://www.marriott.com/",
        "Artech Infosystems Pvt. Ltd.": "https://www.artech.com/", "Collabera Services Pvt. Ltd.": "https://collabera.com/", "Futur Staffing Solutions Pvt. Ltd.": "https://www.futurstaffing.com/",
        "Sarthee Consultancy": "https://sarthee.com/", "American Chase": "https://www.american-chase.com/", "Anaxee Digital Runners Pvt. Ltd.": "https://anaxee.com/",
        "Cogent Infotech": "https://www.cogentinfo.com/", "Netlink Software Pvt. Ltd.": "https://www.netlink.com/", "Tata Consultancy Services Ltd.": "https://www.tcs.com/",
        "Bhaskar Industries Pvt. Ltd.": "https://www.bhaskar.com/", "Impression Furniture Industries Pvt. Ltd.": "https://www.impressionfurniture.com/", "Motilal Oswal": "https://www.motilaloswal.com/",
        "MPM Ltd.": "https://www.mpm.co.in/", "Shakesteller Energy Solutions Pvt. Ltd.": "https://shakesteller.com/", "Trident Group": "https://www.tridentindia.com/",
        "DB Corp Ltd. (Dainik Bhaskar)": "https://www.dbcorpltd.com/", "The Times Group": "https://timesgroup.com/", "Aditya Capital Pvt. Ltd.": "https://www.adityacapital.com/",
        "Bajaj Finserv Ltd.": "https://www.bajajfinserv.in/", "Bajaj Housing Finance Ltd.": "https://www.bajajhousingfinance.in/", "ICICI Securities Ltd.": "https://www.icicidirect.com/",
        "India Shelter Finance Corporation Ltd.": "https://www.indiashelter.in/", "NJ India Invest Pvt. Ltd.": "https://www.njgroup.in/", "Ashiana Housing Ltd.": "https://www.ashianahousing.com/",
        "MoneyOne Consulting Pvt. Ltd.": "https://moneyone.in/", "SCG Group": "https://www.scggroup.com/", "Deloitte Consulting India Pvt. Ltd.": "https://www.deloitte.com/in/en.html",
        "Aditya Birla Lifestyle Brands Ltd.": "https://www.adityabirlafashion.com/", "Avenue Supermarts (DMart)": "https://www.dmartindia.com/", "Bluestone Jewellery & Lifestyle Ltd.": "https://www.bluestone.com/",
        "Pantaloons Fashion & Retail Ltd.": "https://www.pantaloons.com/", "Parnalan Fashion & Retail Ltd. (Calvin Klein & Tommy Hilfiger)": "https://www.abfrl.com/", "SolarSquare Energy Pvt. Ltd.": "https://solarsquare.in/",
    }

    # Build a searchable profile table while preserving all source entries.
    company_rows = []
    for source_no, raw_name, sector in IPER_2026_COMPANIES:
        clean_name = display_company_name(raw_name)
        base = clean_name
        if base.startswith("Gujarat Cooperative Milk Marketing Federation"):
            base = "Gujarat Cooperative Milk Marketing Federation Ltd. (Amul)"
        if base.startswith("The Times Group"):
            base = "The Times Group"
        if base.startswith("ICICI Prudential Life Insurance"):
            base = "ICICI Prudential Life Insurance Co. Ltd."
        if base.startswith("HDFC Life Insurance"):
            base = "HDFC Life Insurance Co. Ltd."
        if base.startswith("Bajaj Life Insurance"):
            base = "Bajaj Life Insurance Ltd."
        if base.startswith("Home First Finance"):
            base = "Home First Finance Co. India Ltd."
        if base.startswith("KMV Ventures"):
            base = "KMV Ventures Pvt. Ltd."
        if base.startswith("Yash Technologies"):
            base = "Yash Technologies Pvt. Ltd."
        if base.startswith("Toprankers"):
            base = "Toprankers Edtech Solutions Pvt. Ltd."
        if base.startswith("PlanetSpark"):
            base = "PlanetSpark"
        if base.startswith("Motilal Oswal"):
            base = "Motilal Oswal"
        if base.startswith("CarWale"):
            base = "CarWale (CarTrade Tech Ltd.)"
        if base.startswith("Avenue Supermarts"):
            base = "Avenue Supermarts (DMart)"
        if base.startswith("Ashiana Housing"):
            base = "Ashiana Housing Ltd."
        if base.startswith("SolarSquare"):
            base = "SolarSquare Energy Pvt. Ltd."
        if base.startswith("Impression Furniture"):
            base = "Impression Furniture Industries Pvt. Ltd."
        note = COMPANY_NOTES.get(base)
        if not note:
            note = f"IPER's 2025–26 placement material lists this organisation. For preparation, study its {sector.lower()} business model, the role-specific job description and the current priorities of the business."
        company_rows.append({"No": source_no, "Company": clean_name, "Sector": sector, "Note": note, "Website": COMPANY_WEBSITES.get(base)})

    # Student-friendly sector selector
    st.markdown("### Explore Indian Sectors")
    selected_sector = st.selectbox("Select a sector", list(SECTOR_INFO.keys()))
    sector_data = SECTOR_INFO[selected_sector]

    c1, c2 = st.columns([1.65, 1])
    with c1:
        st.markdown(f"#### {selected_sector}")
        st.write(sector_data["about"])
        st.markdown("**Typical MBA Roles**")
        st.info(sector_data["roles"])
        st.markdown("**Skills to Build**")
        st.info(sector_data["skills"])
        st.markdown("**Placement Interview Preparation**")
        st.write(sector_data["prepare"])
    with c2:
        st.markdown("#### IBEF Reference")
        st.write("Use IBEF for the latest sector reports, market context and India-industry developments.")
        st.link_button("Open IBEF sector reference", IBEF_SECTOR_URLS[selected_sector], use_container_width=True)
        st.caption("The portal provides original student-oriented summaries. IBEF is the external reference for sector research.")

    st.markdown("---")
    st.markdown("### Companies Documented in IPER 2025–26")
    st.caption("The list below is transcribed from IPER's published 2025–26 recruiter graphic. Repeated names/entries are retained where they appear in the source; they should not be interpreted as separate companies or as a guarantee of recruitment every year.")

    search_company = st.text_input("Search company or sector", placeholder="e.g. HDFC, Asian Paints, Deloitte, Banking, EdTech")
    sector_filter = st.selectbox("Filter by sector", ["All sectors"] + sorted(SECTOR_INFO.keys()))
    filtered_rows = [
        row for row in company_rows
        if (sector_filter == "All sectors" or row["Sector"] == sector_filter)
        and (not search_company.strip() or search_company.lower() in row["Company"].lower() or search_company.lower() in row["Sector"].lower())
    ]

    st.markdown(f"**{len(filtered_rows)} source entries shown**")
    for row in filtered_rows:
        with st.expander(f"{row['Company']}  ·  {row['Sector']}"):
            left, right = st.columns([1.5, 1])
            with left:
                st.markdown(f"**Sector:** {row['Sector']}")
                st.markdown(f"**IPER source entry:** #{row['No']} — 2025–26")
                st.write(row["Note"])
            with right:
                st.markdown("**About Us / Official Website**")
                if row.get("Website"):
                    st.link_button("Open Company Website ↗", row["Website"], use_container_width=True)
                else:
                    st.caption("Official website link not verified")
                st.markdown("**Prepare for roles such as**")
                st.write(SECTOR_INFO[row["Sector"]]["roles"])
                st.markdown("**Sector reference**")
                st.link_button("Read on IBEF", IBEF_SECTOR_URLS[row["Sector"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### How to Research a Company Before a Placement Drive")
    research_cols = st.columns(4)
    research_steps = [
        ("01", "Business", "What does the company sell or provide? Who are its customers?"),
        ("02", "Industry", "What is changing in the sector? Use IBEF for India-level context."),
        ("03", "Role", "What will the MBA role actually deliver? Read the placement JD carefully."),
        ("04", "Interview", "Prepare company facts, competitors, current developments and role-specific questions."),
    ]
    for col, (num, title, text) in zip(research_cols, research_steps):
        with col:
            st.markdown(f"<div style='border:1px solid #E2E8F0;border-radius:11px;padding:15px;min-height:145px;background:#F8FAFC;'><div style='font-size:12px;font-weight:800;color:#433B86;'>{num}</div><div style='font-size:17px;font-weight:800;color:#0F172A;margin-top:5px;'>{title}</div><div style='font-size:13px;color:#475569;margin-top:7px;line-height:1.5;'>{text}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Official References")
    st.markdown("- **IPER Placements 2026:** [View IPER's official placement page](https://iper.ac.in/placements-2026/)")
    st.markdown("- **IBEF Indian Industries:** [Explore IBEF's industry directory](https://www.ibef.org/index.php/industry.aspx)")
    st.caption("Company presence is sourced from IPER's published placement material. Sector explanations are original placement-preparation content structured with reference to IBEF industry resources. Current roles, openings and recruitment status should always be checked against the official placement notice/JD.")

# SECTION 1: RESUME CHECKER & JOB MATCHER
# SECTION: ABOUT MYSELF
elif selected_nav == "About Myself":
    st.title("About Myself")
    st.caption("Build your personal interview introduction and keep it progressively updated as you add new skills, experiences and achievements.")

    student_id = st.session_state.get("student_id")
    existing_profile = load_student_profile(student_id) if student_id else {}

    fields = {
        "tenth_percentage": st.text_input("10th Percentage", value=existing_profile.get("tenth_percentage", ""), placeholder="e.g., 82%"),
        "twelfth_percentage": st.text_input("12th Percentage", value=existing_profile.get("twelfth_percentage", ""), placeholder="e.g., 78%"),
        "undergraduation": st.text_area("Undergraduation", value=existing_profile.get("undergraduation", ""), placeholder="Degree, college/university, specialization, year, relevant learning", height=90),
        "post_graduation": st.text_area("Post Graduation", value=existing_profile.get("post_graduation", ""), placeholder="MBA/PG degree, specialization, institute, relevant learning", height=90),
        "internship": st.text_area("Internship", value=existing_profile.get("internship", ""), placeholder="Company, role, duration, responsibilities, key learning", height=100),
        "certification_courses": st.text_area("Certification Courses", value=existing_profile.get("certification_courses", ""), placeholder="Courses, certifications, platforms and key skills learned", height=100),
        "activities_participated": st.text_area("Activities Participated", value=existing_profile.get("activities_participated", ""), placeholder="Clubs, events, competitions, sports, volunteering, leadership activities", height=100),
        "achievements": st.text_area("Achievements", value=existing_profile.get("achievements", ""), placeholder="Academic, professional, competition or extracurricular achievements", height=100),
        "professional_interest": st.text_area("Professional Interest", value=existing_profile.get("professional_interest", ""), placeholder="Roles, sectors, functions or career areas you want to pursue", height=100),
    }

    profile_payload = {k: (v or "").strip() for k, v in fields.items()}
    profile_hash = hashlib.sha256(json.dumps(profile_payload, sort_keys=True).encode("utf-8")).hexdigest()
    previous_hash = existing_profile.get("profile_hash", "")

    st.markdown("### Your AI-Generated About Yourself")
    if existing_profile.get("about_yourself"):
        st.markdown(f"<div style='background:#F8FAFC;border:1px solid #CBD5E1;border-radius:10px;padding:18px;line-height:1.75;'>{existing_profile['about_yourself'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        if existing_profile.get("updated_at"):
            st.caption(f"Last revised: {existing_profile['updated_at']}")
    else:
        st.info("Complete your profile and click Generate / Revise About Yourself. Your introduction will be stored for future sessions.")

    changed = profile_hash != previous_hash
    col1, col2 = st.columns([1, 1])
    with col1:
        generate_clicked = st.button("Generate / Revise About Yourself", use_container_width=True)
    with col2:
        st.markdown("**Progressive profile:** Add a new skill, certification, achievement or experience and generate again. The AI will revise the complete introduction using your latest profile.")

    if generate_clicked or (changed and any(profile_payload.values()) and not existing_profile.get("about_yourself")):
        with st.spinner("Building your personalized About Yourself response..."):
            about = generate_about_yourself(profile_payload)
        save_student_profile(student_id, profile_payload, about, profile_hash)
        st.success("Your About Yourself has been updated with your latest profile details.")
        st.rerun()

    st.markdown("### Recommended Interview Structure")
    st.write("Education → Internship / practical exposure → Certifications & skills → Activities & achievements → Professional interest → Career direction")

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
    tabs=st.tabs(["Core Placement Questions","Subject & Company Preparation","What Did I Learn?"])
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
                            "Question": st.session_state.get("current_question", ""),
                            "DurationSeconds": float(st.session_state.get("communication_duration", 0.0) or 0.0)
                        }
                        st.session_state["history"].append(attempt)
                        save_student_attempt(
                            st.session_state["student_id"],
                            category,
                            score,
                            mode,
                            st.session_state.get("current_question", ""),
                            st.session_state.get("communication_duration", 0.0)
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
