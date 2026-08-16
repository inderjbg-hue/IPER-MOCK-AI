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
import uuid
from datetime import datetime

DATABASE_PATH = os.getenv("DATABASE_PATH", "students.db")
ALLOWED_EMAIL_DOMAIN = "@iper.ac.in"


def is_valid_iper_email(email):
    email = email.strip().lower()
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@iper\.ac\.in", email))


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

# ------------------------------------------------------------------------------
# 5B. GROUP DISCUSSION DATABASE & CONTENT
# ------------------------------------------------------------------------------
def init_gd_database():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gd_rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_code TEXT NOT NULL UNIQUE,
            topic TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Waiting',
            started_at TEXT,
            ended_at TEXT,
            max_participants INTEGER NOT NULL DEFAULT 6
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gd_participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            joined_at TEXT NOT NULL,
            left_at TEXT,
            UNIQUE(room_id, student_id),
            FOREIGN KEY(room_id) REFERENCES gd_rooms(id),
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gd_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            self_rating INTEGER NOT NULL,
            confidence_rating INTEGER NOT NULL,
            listening_rating INTEGER NOT NULL,
            teamwork_rating INTEGER NOT NULL,
            key_strength TEXT,
            improvement_area TEXT,
            submitted_at TEXT NOT NULL,
            UNIQUE(room_id, student_id),
            FOREIGN KEY(room_id) REFERENCES gd_rooms(id),
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    conn.commit()
    conn.close()


def generate_gd_room_code():
    conn = get_db_connection()
    try:
        for _ in range(20):
            code = secrets.token_hex(3).upper()
            exists = conn.execute("SELECT 1 FROM gd_rooms WHERE room_code = ?", (code,)).fetchone()
            if not exists:
                return code
        raise RuntimeError("Could not generate a unique GD room code.")
    finally:
        conn.close()


def create_gd_room(topic, created_by):
    code = generate_gd_room_code()
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO gd_rooms(room_code, topic, created_by, created_at, status, max_participants) VALUES (?, ?, ?, ?, 'Waiting', 6)",
        (code, topic, created_by, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
    return code


def get_gd_room(room_code):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM gd_rooms WHERE room_code = ?", (room_code.strip().upper(),)).fetchone()
    conn.close()
    return row


def get_gd_participants(room_id):
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT p.*, s.first_name, s.last_name, s.scholar_id, s.email
        FROM gd_participants p
        JOIN students s ON s.id = p.student_id
        WHERE p.room_id = ?
        ORDER BY p.joined_at
    """, (room_id,)).fetchall()
    conn.close()
    return rows


def join_gd_room(room_code, student_id):
    room = get_gd_room(room_code)
    if not room:
        return False, "Invalid GD room code."
    if room["status"] == "Finished":
        return False, "This GD has already been finished."

    conn = get_db_connection()
    existing = conn.execute(
        "SELECT id FROM gd_participants WHERE room_id = ? AND student_id = ?",
        (room["id"], student_id)
    ).fetchone()
    if existing:
        conn.close()
        return True, "You are already registered in this GD room."

    count = conn.execute(
        "SELECT COUNT(*) AS c FROM gd_participants WHERE room_id = ?", (room["id"],)
    ).fetchone()["c"]
    if count >= room["max_participants"]:
        conn.close()
        return False, "This GD room is full. Maximum capacity is 6 students."
    if room["status"] == "Started":
        conn.close()
        return False, "The GD has already started. New participants cannot join."

    conn.execute(
        "INSERT INTO gd_participants(room_id, student_id, joined_at) VALUES (?, ?, ?)",
        (room["id"], student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
    return True, "You have joined the GD room."


def start_gd_room(room_id):
    conn = get_db_connection()
    conn.execute(
        "UPDATE gd_rooms SET status='Started', started_at=? WHERE id=? AND status='Waiting'",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), room_id)
    )
    conn.commit()
    conn.close()


def finish_gd_room(room_id):
    conn = get_db_connection()
    conn.execute(
        "UPDATE gd_rooms SET status='Finished', ended_at=? WHERE id=? AND status != 'Finished'",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), room_id)
    )
    conn.commit()
    conn.close()


def save_gd_feedback(room_id, student_id, self_rating, confidence_rating, listening_rating, teamwork_rating, key_strength, improvement_area):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO gd_feedback(
            room_id, student_id, self_rating, confidence_rating, listening_rating,
            teamwork_rating, key_strength, improvement_area, submitted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(room_id, student_id) DO UPDATE SET
            self_rating=excluded.self_rating,
            confidence_rating=excluded.confidence_rating,
            listening_rating=excluded.listening_rating,
            teamwork_rating=excluded.teamwork_rating,
            key_strength=excluded.key_strength,
            improvement_area=excluded.improvement_area,
            submitted_at=excluded.submitted_at
    """, (room_id, student_id, self_rating, confidence_rating, listening_rating, teamwork_rating, key_strength, improvement_area, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


GD_DOS = [
    "Understand the topic quickly and define it before taking a position.",
    "Speak briefly but add substance, examples, data, or reasoning.",
    "Listen actively and build on relevant points made by others.",
    "Use a calm, professional tone and maintain respectful disagreement.",
    "Try to bring quieter participants into the discussion.",
    "Use structure: point → reason → example → implication.",
    "Summarise the group's major arguments when appropriate.",
    "Focus on the group objective rather than trying to 'win' the discussion.",
]

GD_DONTS = [
    "Do not dominate the discussion or interrupt repeatedly.",
    "Do not attack a person because you disagree with an idea.",
    "Do not invent statistics or present uncertain claims as facts.",
    "Do not repeat the same point without adding value.",
    "Do not stay silent throughout the discussion.",
    "Do not turn the GD into a one-to-one argument.",
    "Do not use slang, offensive language, or an aggressive tone.",
    "Do not rush to conclude without considering other perspectives.",
]

GD_TOPICS = [
    "Artificial Intelligence: Job Creator or Job Destroyer?",
    "Should AI-generated content be regulated?",
    "Will AI replace entry-level corporate jobs?",
    "The future of work: Remote, hybrid, or office?",
    "Four-day work week: Productivity boost or business risk?",
    "Is work-life balance more important than salary?",
    "Should companies monitor employees digitally?",
    "Skill-based hiring versus degree-based hiring",
    "Is an MBA still worth it in the age of AI?",
    "Internships: Learning opportunity or cheap labour?",
    "Social media: Empowerment or addiction?",
    "Influencer marketing: Trustworthy or manipulative?",
    "Should social media platforms be responsible for misinformation?",
    "Short-form content is reducing attention spans",
    "Personal branding: Essential for careers?",
    "Data privacy versus personalised digital services",
    "Should governments regulate big technology companies?",
    "Digital payments: Is a cashless economy realistic?",
    "Cybersecurity is everyone's responsibility",
    "Can India become a global technology leader?",
    "Startup culture: Innovation versus hype",
    "Unicorns: Real value or inflated valuations?",
    "Bootstrapping versus venture capital",
    "Profit versus purpose in modern business",
    "Are family businesses better than professionally managed firms?",
    "Customer experience is more important than product quality",
    "Brand loyalty is disappearing",
    "Price wars: Good for consumers, bad for brands?",
    "Is advertising becoming too intrusive?",
    "Sustainability as strategy versus sustainability as marketing",
    "Greenwashing: How should companies respond?",
    "Should businesses prioritise ESG over short-term profit?",
    "Electric vehicles: Revolution or transition challenge?",
    "Renewable energy and India's economic future",
    "Climate change: Individual responsibility or corporate responsibility?",
    "Plastic bans: Effective solution or incomplete approach?",
    "Sustainable fashion: Trend or necessity?",
    "Fast fashion versus ethical fashion",
    "Can economic growth and environmental protection coexist?",
    "Water scarcity and India's future",
    "Is urbanisation helping or hurting India?",
    "Smart cities: Technology or basic infrastructure first?",
    "Public transport versus private vehicles",
    "Should cities restrict private car ownership?",
    "Tourism: Economic opportunity or environmental burden?",
    "Can India become a global tourism hub?",
    "Workplace diversity improves business performance",
    "Gender diversity: Business necessity or social responsibility?",
    "Should companies publish salary ranges?",
    "Equal pay: Policy versus implementation",
    "Mental well-being at work: Employer responsibility?",
    "Employee loyalty in the gig economy",
    "Gig work: Freedom or insecurity?",
    "Should gig workers receive employee benefits?",
    "Leadership: Born or developed?",
    "Is empathy a leadership strength?",
    "Should leaders be transparent about business failures?",
    "Autocratic versus democratic leadership",
    "Can a good manager always be a good leader?",
    "Team performance versus individual performance",
    "Competition within teams: Healthy or harmful?",
    "Conflict at work: Problem or opportunity?",
    "Communication skills versus technical skills",
    "Can communication skills be taught?",
    "Is emotional intelligence more important than IQ at work?",
    "Should companies hire for attitude and train for skills?",
    "Experience versus fresh ideas in hiring",
    "Campus placements: Skills or academic scores?",
    "Should college attendance be compulsory?",
    "Online education versus classroom education",
    "AI in education: Personalised learning or dependency?",
    "Should students be allowed to use AI for assignments?",
    "Examinations versus continuous assessment",
    "Vocational education versus traditional degrees",
    "Financial literacy should be compulsory in college",
    "Is entrepreneurship a better career than employment?",
    "Government jobs versus private sector careers",
    "Should students take a gap year before an MBA?",
    "Is salary the best measure of career success?",
    "India's demographic dividend: Opportunity or challenge?",
    "Can India become a developed economy by 2047?",
    "Make in India: Has it changed India's manufacturing story?",
    "Startup India and the future of entrepreneurship",
    "Atmanirbhar Bharat: Self-reliance or global integration?",
    "India versus China as a manufacturing destination",
    "MSMEs: Backbone of India's economy?",
    "Formalisation of the Indian economy: Benefits and challenges",
    "Inflation: Who should bear the burden?",
    "Cashless economy: Inclusion versus exclusion",
    "Universal basic income: Practical policy or expensive idea?",
    "Should governments subsidise essential goods?",
    "Public-private partnerships: Solution or compromise?",
    "Privatisation: Efficiency versus public interest",
    "Is economic inequality India's biggest challenge?",
    "Consumerism: Growth engine or social problem?",
    "Online shopping versus local retailers",
    "Quick commerce: Convenience versus sustainability",
    "E-commerce will replace traditional retail",
    "Cash on delivery versus digital payments",
    "Should product reviews be regulated?",
    "Customer data is the new currency",
    "Are loyalty programmes still effective?",
    "The future of Indian banking is digital",
    "FinTech versus traditional banking",
    "Should cryptocurrencies be widely adopted?",
    "Financial inclusion through digital banking",
    "Social responsibility versus shareholder responsibility",
    "Can business ethics coexist with aggressive targets?",
    "Whistleblowing: Loyalty versus integrity",
    "Corporate governance failures: Who is responsible?",
    "Transparency builds stronger brands",
    "Is nationalism good for business?",
    "Globalisation versus localisation",
    "Geopolitics and its impact on Indian businesses",
    "Should companies diversify supply chains even at higher cost?",
    "India's role in the global economy",
    "Sports as a business: Opportunity versus commercialisation",
    "Women in sports and equal opportunities",
    "Is celebrity endorsement still effective?",
    "News media: Business model versus public service",
    "Freedom of expression versus responsible communication",
]

# Keep exactly 100+ topics available even if the list is later edited.
GD_TOPICS = GD_TOPICS[:120]


def generate_gd_perspectives(topic):
    prompt = f"""
Act as a neutral MBA Group Discussion mentor at IPER Bhopal.
The student is preparing for a GD on this topic:

{topic}

Provide guidance only, not a scripted speech. Give: 
1. What the topic means / key context.
2. 5 strong arguments supporting one side.
3. 5 strong counterarguments / opposing perspectives.
4. Economic/business perspective.
5. Social perspective.
6. Technology/future perspective where relevant.
7. Indian context and examples where appropriate.
8. Stakeholder perspectives.
9. 5 questions a participant can use to deepen the discussion.
10. A balanced 4-5 line conclusion.
Avoid fabricated statistics. Clearly label examples as examples when exact data is not certain.
"""
    return get_groq_response(prompt)


init_gd_database()

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
        "Interview Preparation Guide", 
        "Interview Practice Room", 
        "Group Discussion",
        "Mentor Console",
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

# SECTION 4: GROUP DISCUSSION
elif selected_nav == "Group Discussion":
    st.title("Group Discussion (GD) Hub")
    st.caption(f"Build GD knowledge, explore perspectives, and practise with a mentor-generated virtual GD room, {st.session_state.get('first_name', 'Student') }.")

    gd_prep_tab, gd_practice_tab = st.tabs(["📚 GD Preparation", "🎥 GD Practice Room"])

    with gd_prep_tab:
        st.subheader("GD Preparation")
        prep_col1, prep_col2 = st.columns(2)
        with prep_col1:
            st.markdown("### Do's")
            for item in GD_DOS:
                st.markdown(f"- {item}")
        with prep_col2:
            st.markdown("### Don'ts")
            for item in GD_DONTS:
                st.markdown(f"- {item}")

        st.markdown("---")
        st.subheader(f"GD Topic Bank — {len(GD_TOPICS)} Topics")
        topic = st.selectbox("Select a topic for AI-guided perspectives:", GD_TOPICS)
        if st.button("Explore All Perspectives with AI", use_container_width=True):
            with st.spinner("Preparing balanced perspectives for your GD topic..."):
                perspective = generate_gd_perspectives(topic)
            st.markdown(perspective)
            st.info("Use this as preparation guidance. Do not memorise it word-for-word; build your own points and respond to what others say.")

    with gd_practice_tab:
        st.subheader("Virtual GD Practice")
        st.write("Your mentor creates a room code. A maximum of **6 students** can register in each GD room using their IPER student accounts.")

        if "gd_room_code" not in st.session_state:
            st.session_state["gd_room_code"] = ""

        join_col1, join_col2 = st.columns([2, 1])
        with join_col1:
            room_code_input = st.text_input("Enter Mentor-Generated GD Code", value=st.session_state.get("gd_room_code", ""), max_chars=6, placeholder="e.g. A4F91C")
        with join_col2:
            st.write("")
            st.write("")
            if st.button("Join GD Room", use_container_width=True):
                ok, message = join_gd_room(room_code_input, st.session_state["student_id"])
                if ok:
                    st.session_state["gd_room_code"] = room_code_input.strip().upper()
                    st.success(message)
                else:
                    st.error(message)

        active_code = st.session_state.get("gd_room_code", "").strip().upper()
        if active_code:
            room = get_gd_room(active_code)
            if room:
                participants = get_gd_participants(room["id"])
                st.markdown("---")
                room_col1, room_col2, room_col3 = st.columns(3)
                room_col1.metric("Room Code", room["room_code"])
                room_col2.metric("Participants", f"{len(participants)} / 6")
                room_col3.metric("Status", room["status"])

                st.markdown(f"### Topic: {room['topic']}")
                st.markdown("**Students in the room**")
                for p_row in participants:
                    st.write(f"• {p_row['first_name']} {p_row['last_name']} — Scholar ID: `{p_row['scholar_id']}`")

                if room["status"] == "Waiting":
                    if any(p_row["student_id"] == st.session_state["student_id"] for p_row in participants):
                        if st.button("Start GD", use_container_width=True):
                            start_gd_room(room["id"])
                            st.rerun()
                        st.info("The GD can start when the participants are ready. Maximum capacity is 6 students.")
                    else:
                        st.warning("Join this room before trying to start the GD.")
                elif room["status"] == "Started":
                    st.success("GD is live. Join the virtual meeting below and use your real name + Scholar ID when prompted by the meeting room.")
                    jitsi_url = f"https://meet.jit.si/IPER-GD-{room['room_code']}"
                    st.markdown(f"**Virtual GD link:** {jitsi_url}")
                    components.iframe(jitsi_url, height=650, scrolling=True)
                    if st.button("Finish GD", use_container_width=True):
                        finish_gd_room(room["id"])
                        st.rerun()
                else:
                    st.info("This GD has finished. Please submit your self-feedback below.")

                if room["status"] == "Finished":
                    st.markdown("---")
                    st.subheader("Your GD Self-Feedback")
                    st.caption("Reflect on your own performance. Your feedback is saved against this GD room.")
                    with st.form(f"gd_feedback_form_{room['id']}"):
                        self_rating = st.slider("Overall performance", 1, 5, 3)
                        confidence_rating = st.slider("Confidence", 1, 5, 3)
                        listening_rating = st.slider("Listening & building on others' ideas", 1, 5, 3)
                        teamwork_rating = st.slider("Teamwork & respectful participation", 1, 5, 3)
                        key_strength = st.text_area("What was your biggest strength in this GD?", max_chars=500)
                        improvement_area = st.text_area("What will you improve in your next GD?", max_chars=500)
                        submitted = st.form_submit_button("Save My GD Feedback", use_container_width=True)
                    if submitted:
                        save_gd_feedback(room["id"], st.session_state["student_id"], self_rating, confidence_rating, listening_rating, teamwork_rating, key_strength, improvement_area)
                        st.success("Your GD feedback has been saved successfully.")
            else:
                st.error("That GD room no longer exists. Please ask your mentor for a new code.")

# SECTION 5: MENTOR CONSOLE
elif selected_nav == "Mentor Console":
    st.title("Mentor Console")
    st.caption("Create and manage mentor-generated GD rooms for up to 6 IPER students.")

    mentor_key = st.secrets.get("MENTOR_ACCESS_KEY", os.getenv("MENTOR_ACCESS_KEY", ""))
    if not mentor_key:
        st.warning("Mentor access is not configured. Add `MENTOR_ACCESS_KEY` to Streamlit Secrets or the deployment environment.")
    else:
        entered_key = st.text_input("Mentor Access Key", type="password")
        if entered_key == mentor_key:
            st.success("Mentor access verified.")
            mentor_topic = st.selectbox("Select GD Topic", GD_TOPICS + ["Custom Topic"])
            if mentor_topic == "Custom Topic":
                mentor_topic = st.text_input("Enter Custom GD Topic")

            if st.button("Generate GD Room Code", use_container_width=True):
                if not mentor_topic.strip():
                    st.error("Please select or enter a GD topic.")
                else:
                    code = create_gd_room(mentor_topic.strip(), "Mentor")
                    st.session_state["mentor_last_code"] = code
                    st.success(f"GD room created. Share this code with up to 6 students: {code}")

            if st.session_state.get("mentor_last_code"):
                code = st.session_state["mentor_last_code"]
                room = get_gd_room(code)
                if room:
                    st.markdown("---")
                    st.subheader(f"Active Room: {code}")
                    st.write(f"**Topic:** {room['topic']}")
                    participants = get_gd_participants(room["id"])
                    st.metric("Registered Students", f"{len(participants)} / 6")
                    for p_row in participants:
                        st.write(f"• {p_row['first_name']} {p_row['last_name']} — `{p_row['scholar_id']}`")

                    if room["status"] == "Waiting":
                        if st.button("Start GD for This Room", use_container_width=True):
                            start_gd_room(room["id"])
                            st.rerun()
                    elif room["status"] == "Started":
                        st.success("GD is currently live.")
                        if st.button("End GD", use_container_width=True):
                            finish_gd_room(room["id"])
                            st.rerun()
                    else:
                        st.info("GD finished. Students can now submit their self-feedback.")
        elif entered_key:
            st.error("Invalid mentor access key.")

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
