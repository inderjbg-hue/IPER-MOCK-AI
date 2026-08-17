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

GROQ_API_KEY = None
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
elif os.getenv("GROQ_API_KEY"):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

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

def transcribe_indian_english_audio(audio_path):
    prompt = "This is an official MBA placement interview response in Indian English at IPER Bhopal."
    result = whisper_model.transcribe(
        audio_path,
        language="en",
        initial_prompt=prompt,
        temperature=0.0
    )
    return result.get("text", "").strip()

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
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Execution Error: {str(e)}"

def render_video_recorder_component():
    html_code = """
    <div style="font-family: 'Inter', sans-serif; border: 1px solid #CBD5E1; border-radius: 6px; padding: 15px; background: #FFFFFF;">
        <video id="preview" autoplay playsinline muted style="width: 100%; max-height: 240px; background: #000; border-radius: 4px;"></video>
        <div style="margin-top: 12px; display: flex; gap: 10px;">
            <button id="startBtn" onclick="startRec()" style="background: #0F172A; color: white; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: 600; font-family: 'Inter', sans-serif;">Start Video Recording</button>
            <button id="stopBtn" onclick="stopRec()" disabled style="background: #475569; color: white; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: 600; font-family: 'Inter', sans-serif;">Stop & Save</button>
            <a id="downloadAnchor" style="display:none; background: #0F172A; color: white; text-decoration: none; padding: 8px 14px; border-radius: 4px; font-size: 13px; font-weight: 600; font-family: 'Inter', sans-serif;">Download Video (.webm)</a>
        </div>
        <div id="status" style="margin-top: 10px; font-size: 13px; color: #0F172A; font-weight: 500;">Status: Camera Ready</div>
    </div>
    <script>
        let recorder, chunks = [], streamRef;
        async function startRec() {
            try {
                chunks = [];
                document.getElementById('downloadAnchor').style.display = 'none';
                streamRef = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
                document.getElementById('preview').srcObject = streamRef;
                let options = { mimeType: 'video/webm;codecs=vp9,opus' };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) options = { mimeType: 'video/webm' };
                recorder = new MediaRecorder(streamRef, options);
                recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
                recorder.onstop = () => {
                    const blob = new Blob(chunks, { type: 'video/webm' });
                    const downloadBtn = document.getElementById('downloadAnchor');
                    downloadBtn.href = URL.createObjectURL(blob);
                    downloadBtn.download = 'iper_interview_' + Date.now() + '.webm';
                    downloadBtn.style.display = 'inline-block';
                };
                recorder.start(1000);
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('status').innerText = 'Status: Recording in Progress...';
            } catch (err) {
                document.getElementById('status').innerText = 'Camera Error: ' + err.message;
            }
        }
        function stopRec() {
            if (recorder && recorder.state !== 'inactive') recorder.stop();
            if(streamRef) streamRef.getTracks().forEach(track => track.stop());
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').innerText = 'Status: Recording completed. Download file below.';
        }
    </script>
    """
    components.html(html_code, height=340)

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
        "15. What motivates you to perform at your best in a workplace setting?"
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

# ------------------------------------------------------------------------------
# 5. STUDENT AUTHENTICATION & ACCOUNT MANAGEMENT
# ------------------------------------------------------------------------------
import sqlite3
import hashlib
import secrets
import shutil
import base64
import hmac
from datetime import datetime

DATABASE_PATH = os.getenv("DATABASE_PATH", "students.db")
ALLOWED_EMAIL_DOMAIN = "@iper.ac.in"
GD_ENGINE_URL = os.getenv("GD_ENGINE_URL", st.secrets.get("GD_ENGINE_URL", "http://localhost:8000"))
GD_ENGINE_SECRET = os.getenv("GD_ENGINE_SECRET", st.secrets.get("GD_ENGINE_SECRET", "CHANGE_THIS_SECRET"))


def is_valid_iper_email(email):
    email = email.strip().lower()
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@iper\.ac\.in", email))

def generate_portal_session_key():
    """Create a fresh non-reusable key for every successful portal login."""
    return secrets.token_urlsafe(18)


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db_connection()
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
            recording_status TEXT DEFAULT 'Not Started',
            host_scholar_id TEXT
        )
    """)
    # Safe migration for an older students.db created by the previous version.
    room_columns = {row[1] for row in conn.execute("PRAGMA table_info(gd_rooms)").fetchall()}
    if "host_scholar_id" not in room_columns:
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
    conn.execute("INSERT OR IGNORE INTO gd_rooms (slot, code, topic, status, created_at) VALUES (?, NULL, ?, 'Open', ?)",
                 (GD_SLOT, 'Student GD Practice', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()
    ensure_single_gd_room()


def ensure_single_gd_room():
    """Ensure the single student-created GD room always has a shareable code."""
    conn = get_db_connection()
    row = conn.execute("SELECT code, status FROM gd_rooms WHERE slot=?", (GD_SLOT,)).fetchone()
    if row and not row["code"]:
        code = generate_gd_code() if "generate_gd_code" in globals() else None
        if code:
            conn.execute("UPDATE gd_rooms SET code=? WHERE slot=?", (code, GD_SLOT))
            conn.commit()
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
                first_name.strip(),
                last_name.strip(),
                scholar_id,
                email,
                password_hash,
                password_salt,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return True, "Account created successfully. You can now sign in."
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if "email" in message:
            return False, "This email ID is already registered. Please sign in."
        if "scholar_id" in message:
            return False, "This Scholar ID is already registered."
        return False, "This account could not be created because the details already exist."
    finally:
        conn.close()


def authenticate_student(email, password):
    email = email.strip().lower()
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM students WHERE email = ?", (email,)
    ).fetchone()
    conn.close()

    if row and verify_password(password, row["password_hash"], row["password_salt"]):
        return dict(row)
    return None


def load_student_attempts(student_id):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT timestamp, domain, score, mode, question
        FROM interview_attempts
        WHERE student_id = ?
        ORDER BY id ASC
        """,
        (student_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "Timestamp": row["timestamp"],
            "Candidate": "Student",
            "Domain": row["domain"] or "",
            "Score": row["score"] or 0,
            "Mode": row["mode"] or "",
            "Question": row["question"] or "",
        }
        for row in rows
    ]


def save_student_attempt(student_id, domain, score, mode, question):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO interview_attempts
        (student_id, timestamp, domain, score, mode, question)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            student_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            domain,
            int(score),
            mode,
            question,
        ),
    )
    conn.commit()
    conn.close()



GD_MAX_PARTICIPANTS = 7
GD_MAX_MINUTES = 10
GD_ROOM_COUNT = 1
GD_SLOT = 1
GD_DOMAIN = os.getenv("JITSI_DOMAIN", st.secrets.get("JITSI_DOMAIN", "meet.jit.si"))

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
        SET code=?, topic=?, status='Open', created_at=?, started_at=NULL, ended_at=NULL, recording_status='Not Started', host_scholar_id=NULL
        WHERE slot=?
    """, (code, topic, now, slot))
    conn.commit()
    conn.close()
    return code


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
        # The first student becomes the GD host/recording starter.
        # The actual 10-minute session starts only when the host presses Start GD
        # inside the inbuilt WebRTC room.
        if not room["host_scholar_id"]:
            conn.execute("UPDATE gd_rooms SET host_scholar_id=?, status='Open', started_at=NULL WHERE slot=?",
                         (student["scholar_id"], slot))
        conn.commit()
        return True, "You have joined the GD room. The host will start the 10-minute session when everyone is ready."
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


def create_gd_engine_token(room_code, student, expires_seconds=20 * 60):
    payload = {
        "room": room_code.upper(),
        "scholar_id": student["scholar_id"],
        "display_name": f"{student['first_name']} {student['last_name']} ({student['scholar_id']})",
        "email": student.get("email", ""),
        "exp": int(time.time()) + expires_seconds,
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = hmac.new(GD_ENGINE_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{signature}"


def render_inbuilt_gd_room(slot, room_code, student, is_host=False):
    """Render the IPER-owned WebRTC room. No Google Meet/Jitsi/Zoom dependency."""
    token = create_gd_engine_token(room_code, student)
    room_url = f"{GD_ENGINE_URL.rstrip('/')}/room?room={room_code.upper()}&token={token}"
    safe_url = json.dumps(room_url)
    role = "Student Host / Recording Starter" if is_host else "Student Participant"
    html = f"""
    <div style="font-family:Inter,Arial,sans-serif;border:1px solid #CBD5E1;border-radius:10px;overflow:hidden;background:#0F172A;">
      <div style="padding:10px 14px;color:white;background:#0F172A;display:flex;justify-content:space-between;align-items:center;">
        <div><b>IPER Inbuilt Virtual GD Room</b><br><span style="font-size:12px;opacity:.85">{role} • maximum 7 students • 1080p preferred • maximum 10 minutes</span></div>
      </div>
      <iframe src={safe_url} allow="camera; microphone; autoplay; display-capture" style="width:100%;height:780px;border:0;background:#020617;"></iframe>
    </div>
    """
    components.html(html, height=820, scrolling=False)

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


def generate_gd_feedback_ai(student_name, topic, voice, perspective, participation, camera, strengths, improvements):
    prompt = f"""
You are an MBA placement mentor at IPER Bhopal. Provide concise, constructive GD feedback for {student_name}.
Topic: {topic}
Self-ratings (1-10): Voice/Verbal Delivery={voice}, Perspective/Ideas={perspective}, Participation/Team Contribution={participation}, Camera Clarity/Visual Presence={camera}
Student strength reflection: {strengths}
Student improvement reflection: {improvements}
Return exactly four labelled sections: Voice, Perspective, Participation, Camera Clarity, followed by one Overall Next-Step paragraph. Do not invent facts about what the student actually said; base the guidance only on the supplied ratings and reflections.
"""
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

    login_tab, signup_tab = st.tabs(["🔐 Student Sign In", "📝 Student Sign Up"])

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
                    st.session_state["portal_access_key"] = generate_portal_session_key()
                    st.session_state["history"] = load_student_attempts(student["id"])
                    st.session_state["resume_details"] = None
                    st.success(f"Welcome back, {student['first_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid email ID or password.")

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
st.sidebar.markdown("**New Portal Session Key**")
st.sidebar.code(st.session_state.get("portal_access_key", "N/A"), language=None)
st.sidebar.caption("This key is regenerated on every login. Do not share it; it identifies your current portal session.")
if not FFMPEG_PATH:
    st.sidebar.warning("FFmpeg is not installed. Audio/video transcription will not work until FFmpeg is added to the deployment environment.")
st.sidebar.markdown("---")

selected_nav = st.sidebar.radio(
    "MAIN MENU",
    [
        "Resume Checker & Job Matcher", 
        "Interview Preparation Guide", 
        "Interview Practice Room", 
        "Group Discussion Hub",
        "Performance Dashboard"
    ]
)

if st.sidebar.button("Log Out", use_container_width=True):
    for key in ["authenticated", "student_id", "student_email", "first_name", "last_name", "scholar_id", "candidate_name", "resume_details", "current_question", "portal_access_key", "active_gd_slot", "active_gd_code"]:
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

# SECTION 2: INTERVIEW PREPARATION GUIDE
elif selected_nav == "Interview Preparation Guide":
    st.title("Interview Preparation Guide")
    st.caption("Select a business domain and question to study key evaluation objectives, effective response frameworks, and model benchmark answers.")
    
    prep_category = st.selectbox("Select Study Domain:", ["General and Core Skills"] + SPECIALIZATIONS + ["Company Specific"])
    
    selected_question = None
    if prep_category == "Company Specific":
        comp_choice = st.selectbox("Select Target Company:", IPER_RECRUITERS)
        if st.button("Load Top Recruiter Questions"):
            with st.spinner(f"Retrieving top questions for {comp_choice}..."):
                q_prompt = f"Generate 5 popular technical and situational interview questions asked by {comp_choice} during campus hiring."
                st.markdown(get_groq_response(q_prompt))
    else:
        q_list = EXHAUSTIVE_QUESTIONS.get(prep_category, ["Describe a key challenge you faced and how you resolved it."])
        selected_question = st.selectbox(f"Select Question to Study (15+ Available under {prep_category}):", q_list)

    if selected_question:
        st.markdown("---")
        st.subheader(f"Study Guide: {selected_question}")
        
        if st.button("Generate Complete Answer Framework & Model Answer"):
            with st.spinner("Preparing answer breakdown..."):
                study_prompt = f"""
                Act as a senior MBA Placement Advisor at IPER Bhopal.
                Analyze this interview question for candidates:
                
                Question: "{selected_question}"
                Domain: "{prep_category}"

                Provide a clean, structured guide containing:
                1. WHAT INTERVIEWERS LOOK FOR (Key competencies being evaluated).
                2. HOW TO STRUCTURE YOUR ANSWER (Step-by-step structure like STAR or CAR framework).
                3. MODEL BENCHMARK ANSWER (A clear, high-scoring sample response).
                """
                study_guide = get_groq_response(study_prompt)
                st.markdown(study_guide)

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
                        extracted_transcript = transcribe_indian_english_audio(tmp_path)
                    except Exception as err:
                        st.error(f"Speech transcription error: {err}")
                        if not FFMPEG_PATH:
                            st.info("FFmpeg is missing. Add `ffmpeg` to packages.txt (Streamlit Cloud) or install it in your Dockerfile (Cloud Run).")
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                
                if extracted_transcript:
                    st.success("Audio recorded and transcribed successfully.")

        elif mode == "Video Response Mode":
            st.subheader("1. Record & Save Video")
            render_video_recorder_component()
            
            st.subheader("2. Upload Saved Video File")
            uploaded_video = st.file_uploader("Upload recorded video file (.webm, .mp4):", type=["webm", "mp4"])
            
            if uploaded_video is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                video_ext = "mp4" if uploaded_video.name.lower().endswith(".mp4") else "webm"
                saved_video_filename = f"video_{ts}.{video_ext}"
                saved_video_path = os.path.join(VIDEO_STORAGE_DIR, saved_video_filename)
                
                with open(saved_video_path, "wb") as f:
                    f.write(uploaded_video.read())
                
                st.video(saved_video_path)
                st.success(f"Video file saved: `{saved_video_filename}`")
                
                with st.spinner("Transcribing video audio track via Whisper..."):
                    try:
                        extracted_transcript = transcribe_indian_english_audio(saved_video_path)
                        st.success("Video audio transcribed successfully.")
                    except Exception as err:
                        st.error(f"Speech transcription error: {err}")
                        if not FFMPEG_PATH:
                            st.info("FFmpeg is missing. Add `ffmpeg` to packages.txt (Streamlit Cloud) or install it in your Dockerfile (Cloud Run).")

        final_response_text = st.text_area(
            "Answer Transcript (Review or Edit Text Before Submitting):", 
            value=extracted_transcript, 
            height=140
        )

        if st.button("Submit Response for Feedback"):
            if not final_response_text.strip():
                st.error("Please record your audio/video response or write your transcript text first.")
            else:
                with st.spinner("Analyzing your response..."):
                    c_name = st.session_state.get('first_name', 'Student')
                    eval_prompt = f"""
                    Act as an encouraging MBA placement mentor at IPER Bhopal.
                    Evaluate the response for candidate: {c_name}.
                    Address {c_name} warmly by name across feedback areas.

                    Question Asked: {st.session_state['current_question']}
                    Candidate Response: {final_response_text}
                    Resume Context: {json.dumps(st.session_state.get('resume_details', {}))}
                    Practice Mode: {mode}

                    Return response in VALID JSON strictly matching this structure:
                    {{
                        "CandidateName": "{c_name}",
                        "GradingScore": <0-100 integer>,
                        "ExecutiveSummary": "<Warm greeting addressing {c_name} with an overall summary>",
                        "TechnicalAssessment": "<Evaluation of subject knowledge and key points, addressing {c_name}>",
                        "CommunicationAssessment": "<Assessment of clarity, structure, and delivery, addressing {c_name}>",
                        "ResumeAlignment": "<How effectively {c_name} referenced their experience>",
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
    st.caption("One shared virtual GD room • maximum 7 students • maximum 10 minutes • 1080p preferred")

    gd_prep_tab, gd_practice_tab, gd_feedback_tab = st.tabs(["📚 GD Preparation", "🎥 GD Practice", "📝 My GD Feedback"])

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
        room = get_gd_rooms()[0]
        st.markdown("### 🎥 One Virtual GD Room — 7 Video Participants")
        st.info("Any logged-in IPER student can open the room. The first student becomes the room host and can share the GD Room Code with up to 6 other logged-in students.")

        st.markdown("#### 🔑 GD Room Access Code")
        st.code(room["code"], language=None)
        st.caption("Share ONLY this GD Room Access Code with your six peers. Each participant must still sign in with their own @iper.ac.in student account so the room displays their profile name and Scholar ID.")

        if room.get("status") == "Ended":
            st.warning("The previous GD has ended. Start a fresh session to generate a new room code.")
            if st.button("Create New GD Session", use_container_width=True):
                new_code = reset_gd_slot(GD_SLOT, room.get("topic") or "Student GD Practice")
                st.session_state.pop("active_gd_slot", None)
                st.session_state.pop("active_gd_code", None)
                st.success(f"New GD Room Code: {new_code}")
                st.rerun()

        topic_for_room = st.selectbox("GD Topic for this session", GD_TOPICS, index=GD_TOPICS.index(room["topic"]) if room.get("topic") in GD_TOPICS else 0)
        if room.get("status") == "Open" and st.button("Update Topic Before Starting", use_container_width=True):
            conn = get_db_connection()
            conn.execute("UPDATE gd_rooms SET topic=? WHERE slot=? AND status='Open'", (topic_for_room, GD_SLOT))
            conn.commit()
            conn.close()
            st.success("GD topic updated.")
            st.rerun()

        participants = get_gd_participants(GD_SLOT)
        st.write(f"**Participants:** {len(participants)}/{GD_MAX_PARTICIPANTS} | **Status:** {room['status']}")
        if participants:
            st.markdown("**Joined students:** " + " • ".join([f"{p['first_name']} {p['last_name']} ({p['scholar_id']})" for p in participants]))

        join_code = st.text_input("Enter the shared GD Room Code", value=room["code"], max_chars=6, placeholder="e.g. A4F91C").strip().upper()
        if st.button("Join Virtual GD Room", use_container_width=True):
            ok, message = join_gd_room(GD_SLOT, join_code, {
                "scholar_id": st.session_state["scholar_id"],
                "first_name": st.session_state["first_name"],
                "last_name": st.session_state["last_name"],
                "email": st.session_state["student_email"]
            })
            if ok:
                st.session_state["active_gd_slot"] = GD_SLOT
                st.session_state["active_gd_code"] = join_code
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        active_code = st.session_state.get("active_gd_code")
        active_room = next((r for r in get_gd_rooms() if r["slot"] == GD_SLOT and r.get("code") == active_code), None) if active_code else None
        if active_room:
            participants = get_gd_participants(GD_SLOT)
            is_host = active_room.get("host_scholar_id") == st.session_state["scholar_id"]
            st.markdown("---")
            st.markdown(f"### Live GD: {active_room['topic']}")
            st.write(f"**Room Code:** `{active_room['code']}` | **Participants:** {len(participants)}/{GD_MAX_PARTICIPANTS} | **Your role:** {'Host / Recording Starter' if is_host else 'Participant'}")
            render_inbuilt_gd_room(GD_SLOT, active_room["code"], {
                "first_name": st.session_state["first_name"],
                "last_name": st.session_state["last_name"],
                "scholar_id": st.session_state["scholar_id"],
                "email": st.session_state["student_email"]
            }, is_host=is_host)
            st.warning("The room is limited by the application to 7 registered students. The GD clock starts when the host presses Start GD and is capped at 10 minutes. 1080p is requested; actual camera quality depends on each device, browser, network and server capacity.")
            if is_host and active_room.get("status") == "Active":
                if st.button("End GD Session", type="primary", use_container_width=True):
                    set_gd_status(GD_SLOT, "Ended", "Ended by student host")
                    st.success("GD session ended. Students can now submit their feedback.")
                    st.session_state.pop("active_gd_slot", None)
                    st.session_state.pop("active_gd_code", None)
                    st.rerun()
            if st.button("Leave GD Room", use_container_width=True):
                leave_gd_room(GD_SLOT, st.session_state["scholar_id"])
                st.session_state.pop("active_gd_slot", None)
                st.session_state.pop("active_gd_code", None)
                st.rerun()

    with gd_feedback_tab:
        st.markdown("### Self-Feedback After Your GD")
        st.caption("Rate your own performance on voice, perspective, participation and camera clarity.")
        feedback_rooms = [r for r in get_gd_rooms() if r.get("status") == "Ended"]
        if feedback_rooms:
            feedback_room = feedback_rooms[0]
            st.info(f"Completed GD: **{feedback_room['topic']}** | Room Code: `{feedback_room['code']}`")
            with st.form("gd_feedback_single_room"):
                voice = st.slider("Voice / Verbal Delivery", 1, 10, 7)
                perspective = st.slider("Perspective / Quality of Ideas", 1, 10, 7)
                participation = st.slider("Participation / Listening / Team Contribution", 1, 10, 7)
                camera = st.slider("Camera Clarity / Visual Presence", 1, 10, 7)
                strengths = st.text_area("What was your biggest strength?")
                improvements = st.text_area("What will you improve in your next GD?")
                submitted = st.form_submit_button("Save GD Feedback", use_container_width=True)
                if submitted:
                    save_gd_feedback(GD_SLOT, {"scholar_id": st.session_state["scholar_id"]}, voice, perspective, participation, camera, strengths, improvements)
                    st.success("Your GD feedback has been saved to your profile.")
                    with st.spinner("Generating personalized GD coaching feedback..."):
                        st.markdown("### AI GD Coaching Feedback")
                        st.markdown(generate_gd_feedback_ai(st.session_state.get("first_name", "Student"), feedback_room["topic"], voice, perspective, participation, camera, strengths, improvements))
        else:
            st.info("Complete the shared GD session first. The feedback tab will open after the host ends it.")

        previous_feedback = get_student_gd_feedback(st.session_state["scholar_id"])
        if previous_feedback:
            st.markdown("### Previous GD Feedback")
            for fb in previous_feedback[:10]:
                st.markdown(f"**{fb['created_at']} — Shared GD Room**")
                st.write(f"Voice: {fb['voice_score']}/10 | Perspective: {fb['perspective_score']}/10 | Participation: {fb['participation_score']}/10 | Camera: {fb['camera_clarity_score']}/10")
                if fb.get("strengths"): st.write(f"**Strength:** {fb['strengths']}")
                if fb.get("improvements"): st.write(f"**Improve:** {fb['improvements']}")

# SECTION 6: PERFORMANCE DASHBOARD
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
