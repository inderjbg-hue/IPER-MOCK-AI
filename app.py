import os
import ssl
import tempfile
import time
import json
import re
import pandas as pd
import pypdf
import whisper
from groq import Groq
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------------------------
# 1. STREAMLIT INITIAL CONFIG & SYSTEM PATHS
# ------------------------------------------------------------------------------
st.set_page_config(page_title="IPER MOCK AI", layout="wide")

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" + os.pathsep + "/usr/local/bin"

VIDEO_STORAGE_DIR = "saved_videos"
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. BULLETPROOF CSS (PREVENTS MATERIAL ICON BLEED & DARK OVERLAY BOXES)
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        /* Base Background */
        .main .block-container, .stApp {
            background-color: #FFFFFF !important;
            font-family: 'Inter', sans-serif;
            color: #0F172A !important;
        }

        /* Targeted Text Color (Prevents breaking Material Icons like arrow_drop_down) */
        p, h1, h2, h3, h4, h5, h6, label, span, li, td, th {
            color: #0F172A !important;
        }

        /* Markdown Container Visibility */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #0F172A !important;
        }

        /* Expander Container Fix */
        div[data-aria-expanded="true"] p, 
        div[data-aria-expanded="false"] p,
        [data-testid="stExpander"] details summary p {
            color: #0F172A !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stExpander"] {
            border: 1px solid #E2E8F0 !important;
            background-color: #F8FAFC !important;
            border-radius: 8px !important;
        }

        /* File Uploader Fix (Removes dark overlay box) */
        [data-testid="stFileUploader"] {
            border: 2px dashed #0284C7 !important;
            border-radius: 12px !important;
            padding: 12px !important;
            background-color: #F0F9FF !important;
        }
        
        [data-testid="stFileUploaderDropzone"] {
            background-color: #FFFFFF !important;
        }

        [data-testid="stFileUploader"] * {
            color: #0F172A !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        /* Form Inputs */
        div[data-baseweb="select"] > div,
        .stSelectbox select, 
        .stTextArea textarea, 
        .stTextInput input {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }

        /* Primary Action Buttons */
        .stButton > button {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.6rem 1.3rem !important;
        }
        .stButton > button p {
            color: #FFFFFF !important;
        }
        .stButton > button:hover {
            background-color: #1D4ED8 !important;
        }

        /* Tab Navigation Bar */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 2px solid #E2E8F0;
        }
        
        .stTabs [data-baseweb="tab"]:nth-child(1) {
            background-color: #E0F2FE !important;
            color: #0369A1 !important;
            border-radius: 8px 8px 0 0 !important;
            font-weight: 700 !important;
            padding: 10px 20px !important;
        }
        .stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
        }
        .stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] p {
            color: #FFFFFF !important;
        }
        
        .stTabs [data-baseweb="tab"]:nth-child(2) {
            background-color: #D1FAE5 !important;
            color: #047857 !important;
            border-radius: 8px 8px 0 0 !important;
            font-weight: 700 !important;
            padding: 10px 20px !important;
        }
        .stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
            background-color: #059669 !important;
            color: #FFFFFF !important;
        }
        .stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] p {
            color: #FFFFFF !important;
        }
        
        .stTabs [data-baseweb="tab"]:nth-child(3) {
            background-color: #FEF3C7 !important;
            color: #B45309 !important;
            border-radius: 8px 8px 0 0 !important;
            font-weight: 700 !important;
            padding: 10px 20px !important;
        }
        .stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
            background-color: #D97706 !important;
            color: #FFFFFF !important;
        }
        .stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] p {
            color: #FFFFFF !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & API PIPELINE WITH SECRETS RESOLUTION
# ------------------------------------------------------------------------------
@st.cache_resource
def load_speech_model():
    return whisper.load_model("base")

whisper_model = load_speech_model()

# Fetch GROQ API key safely from Streamlit Secrets or Environment
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

def extract_pdf_data(uploaded_file):
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

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
You are a senior MBA Placement Panel Auditor and Industry Mentor at IPER Bhopal.
Address the candidate respectfully by name in all feedback responses.
Evaluate candidate responses with technical rigor.
"""

def get_groq_response(prompt):
    if not client:
        return "GROQ API Key is missing. Please add 'GROQ_API_KEY' to Streamlit App Settings -> Secrets."
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
    <div style="font-family: 'Inter', sans-serif; border: 2px solid #CBD5E1; border-radius: 8px; padding: 15px; background: #FFFFFF;">
        <video id="preview" autoplay playsinline muted style="width: 100%; max-height: 240px; background: #000; border-radius: 6px;"></video>
        <div style="margin-top: 12px; display: flex; gap: 10px;">
            <button id="startBtn" onclick="startRec()" style="background: #059669; color: white; border: none; padding: 9px 16px; border-radius: 6px; cursor: pointer; font-weight: 600;">Start Video Recording</button>
            <button id="stopBtn" onclick="stopRec()" disabled style="background: #DC2626; color: white; border: none; padding: 9px 16px; border-radius: 6px; cursor: pointer; font-weight: 600;">Stop & Save</button>
            <a id="downloadAnchor" style="display:none; background: #0284C7; color: white; text-decoration: none; padding: 9px 16px; border-radius: 6px; font-size: 14px; font-weight: 600;">Download Recording (.webm)</a>
        </div>
        <div id="status" style="margin-top: 10px; font-size: 13px; color: #0F172A; font-weight: 500;">Status: Camera Idle</div>
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
                document.getElementById('status').innerText = 'Status: Live Recording...';
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
# 4. QUESTION DATA REPOSITORIES
# ------------------------------------------------------------------------------
EXHAUSTIVE_QUESTIONS = {
    "General and Core Skills": [
        "1. Walk me through your resume highlighting key academic achievements.",
        "2. What are your top 3 professional strengths and 2 key areas of improvement?",
        "3. Describe a situation where you led a team under a tight deadline.",
        "4. How do you handle constructive criticism from senior managers?",
        "5. Explain your 5-year career blueprint post-MBA."
    ],
    "Marketing": [
        "1. Differentiate between push and pull marketing strategies with industry examples.",
        "2. How do you design a high-converting digital marketing funnel for B2B SaaS?",
        "3. Explain the 7 Ps of Service Marketing in the hospitality sector."
    ],
    "Finance": [
        "1. Explain the 3 main financial statements and how they interconnect.",
        "2. What is Working Capital, and how do you calculate Net Working Capital?",
        "3. Describe the DCF valuation methodology and how to select a Discount Rate."
    ],
    "Human Resource (HR)": [
        "1. Explain the step-by-step SHRM recruitment and selection pipeline.",
        "2. How do you handle workplace conflict between two senior executives?",
        "3. Explain the 360-Degree Performance Appraisal technique."
    ]
}

SPECIALIZATIONS = ["Marketing", "Finance", "Human Resource (HR)", "Banking and Finance", "Tourism and Services Industry"]
IPER_RECRUITERS = ["Amul", "Asian Paints", "HDFC Bank", "ICICI Securities", "Deloitte", "Trident Group", "Berger Paints"]

# ------------------------------------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.title("IPER MOCK AI")
st.sidebar.subheader("Attempt History & Logs")

if not st.session_state["history"]:
    st.sidebar.info("No practice attempts recorded yet.")
else:
    for idx, session in enumerate(reversed(st.session_state["history"]), 1):
        with st.sidebar.expander(f"Attempt #{len(st.session_state['history']) - idx + 1}: {session['Timestamp']}"):
            st.markdown(f"**Candidate:** {session.get('Candidate', 'N/A')}")
            st.markdown(f"**Domain:** {session['Domain']}")
            st.markdown(f"**Score:** {session['Score']}/100")

# ------------------------------------------------------------------------------
# 6. MAIN APPLICATION LAYOUT
# ------------------------------------------------------------------------------
st.title("IPER MOCK AI - Placement Terminal")
st.caption("Official Placement Preparation and Evaluation Platform - IPER Bhopal")

tab_prep, tab_practice, tab_dashboard = st.tabs(["Preparation Tab", "Practice Tab", "Dashboard"])

# TAB 1: PREPARATION TAB
with tab_prep:
    st.header("Preparation Module")
    st.caption("Select a domain and question to view the Objective, Answering Structure, and Benchmark Sample Answer.")
    
    prep_category = st.selectbox("Select Preparation Category:", ["General and Core Skills"] + SPECIALIZATIONS + ["Company Specific"])
    
    selected_question = None
    if prep_category == "Company Specific":
        comp_choice = st.selectbox("Select Target Company:", IPER_RECRUITERS)
        if st.button("Generate Tailored Recruiter Question Set"):
            with st.spinner(f"Compiling questions for {comp_choice}..."):
                q_prompt = f"Generate 5 top technical interview questions asked by {comp_choice} for MBA hires."
                st.markdown(get_groq_response(q_prompt))
    else:
        q_list = EXHAUSTIVE_QUESTIONS.get(prep_category, ["Describe a key challenge you faced and how you resolved it."])
        selected_question = st.selectbox("Select Question to Study:", q_list)

    if selected_question:
        st.markdown("---")
        st.subheader(f"Question Study Guide: {selected_question}")
        
        if st.button("Generate Detailed Objective, Structure & Sample Answer"):
            with st.spinner("Analyzing question framework..."):
                study_prompt = f"""
                Act as a senior MBA Placement Director at IPER Bhopal.
                Analyze the following interview question for students:
                
                Question: "{selected_question}"
                Category: "{prep_category}"

                Provide a structured guide containing:
                1. OBJECTIVE OF ASKING THE QUESTION (What competencies/skills the panel evaluates).
                2. STRUCTURE OF THE ANSWER (Step-by-step framework like STAR, CAR, or 4Ps).
                3. SAMPLE 100/100 BENCHMARK ANSWER (Comprehensive model answer for MBA hires).
                """
                study_guide = get_groq_response(study_prompt)
                st.markdown(study_guide)

# TAB 2: PRACTICE TAB
with tab_practice:
    st.header("Practice & Evaluation Terminal")
    st.subheader("Candidate Resume Profile")
    
    resume_file = st.file_uploader("Upload candidate resume (PDF) to customize interview context:", type=["pdf"])

    if resume_file is not None:
        resume_text = extract_pdf_data(resume_file)
        with st.spinner("Extracting candidate resume details..."):
            parse_prompt = f"""
            Analyze the following resume text and return STRICT JSON with exact keys:
            {{
                "Name": "<Candidate Full Name>",
                "Education": "<Degrees, Institutions, Specializations>",
                "Work_Experience": "<Total years, key roles, companies, and responsibilities>",
                "Key_Skills": "<Technical and functional skills>",
                "Projects_Achievements": "<Key projects, research, or certifications>",
                "Summary": "<2-3 sentence summary>"
            }}
            Do not include commentary or markdown wrapping outside valid JSON.
            Resume Text:
            {resume_text[:3500]}
            """
            parsed_raw = get_groq_response(parse_prompt)
            
            try:
                clean_json = parsed_raw.replace("```json", "").replace("```", "").strip()
                resume_data = json.loads(clean_json)
                c_name = resume_data.get("Name", "Candidate")
            except Exception:
                c_name = "Candidate"
                resume_data = {"Summary": parsed_raw}
            
            st.session_state["candidate_name"] = c_name
            st.session_state["resume_details"] = resume_data
            
            st.success(f"Resume loaded for {st.session_state['candidate_name']}!")
            
            with st.expander("Extracted Resume Insights"):
                if isinstance(resume_data, dict):
                    st.markdown(f"**Candidate Name:** {resume_data.get('Name', 'N/A')}")
                    st.markdown(f"**Education:** {resume_data.get('Education', 'N/A')}")
                    st.markdown(f"**Work Experience:** {resume_data.get('Work_Experience', 'N/A')}")
                    st.markdown(f"**Key Skills:** {resume_data.get('Key_Skills', 'N/A')}")
                    st.markdown(f"**Projects & Achievements:** {resume_data.get('Projects_Achievements', 'N/A')}")
                    st.markdown(f"**Summary:** {resume_data.get('Summary', 'N/A')}")
                else:
                    st.write(resume_data)

    st.markdown("---")
    
    col_mode, col_diff, col_cat = st.columns(3)
    with col_mode:
        mode = st.radio("Practice Mode:", ["Audio Response Mode", "Video Response Mode"])
    with col_diff:
        difficulty = st.selectbox("Difficulty Level:", ["Basic", "Intermediate", "Expert"])
    with col_cat:
        category = st.selectbox("Question Domain:", ["Resume-Based (Tailored)", "General and Core Skills", "Specialization", "Company Specific"])

    selected_spec = None
    selected_comp = None
    if category == "Specialization":
        selected_spec = st.selectbox("Select Track:", SPECIALIZATIONS)
    elif category == "Company Specific":
        selected_comp = st.selectbox("Select Target Recruiter:", IPER_RECRUITERS)

    # GENERATE INTERVIEW QUESTION
    if st.button("Generate Interview Question", use_container_width=True):
        if category == "Resume-Based (Tailored)":
            if not st.session_state.get("resume_details"):
                st.warning("Please upload a candidate resume first to generate resume-tailored questions.")
            else:
                prompt = f"""
                Act as a strict MBA interviewer at IPER Bhopal. 
                Generate ONE highly specific interview question tailored directly to {st.session_state['candidate_name']}'s resume profile:
                {json.dumps(st.session_state['resume_details'])}
                Difficulty: {difficulty}
                Return ONLY the question text directly.
                """
                with st.spinner(f"Generating resume-specific question for {st.session_state['candidate_name']}..."):
                    st.session_state["current_question"] = get_groq_response(prompt)
        else:
            ctx = f"Domain: {category}, Difficulty: {difficulty}"
            if selected_spec: ctx += f", Specialization: {selected_spec}"
            if selected_comp: ctx += f", Target Company: {selected_comp}"
            if st.session_state["resume_details"]: ctx += f", Resume Details: {json.dumps(st.session_state['resume_details'])}"
            
            prompt = f"Act as a strict MBA interviewer at IPER Bhopal. Generate ONE interview question for candidate '{st.session_state['candidate_name']}' based on context: {ctx}. Return ONLY the question text directly."
            with st.spinner("Retrieving question..."):
                st.session_state["current_question"] = get_groq_response(prompt)

    if "current_question" in st.session_state:
        st.info(f"**Assigned Question for {st.session_state['candidate_name']}:** {st.session_state['current_question']}")
        
        extracted_transcript = ""
        saved_video_filename = "N/A"

        if mode == "Audio Response Mode":
            st.subheader("Record Audio Answer")
            audio_data = st.audio_input("Record your answer:")
            if audio_data is not None:
                st.audio(audio_data)
                with st.spinner("Transcribing audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_data.getvalue())
                        tmp_path = tmp_file.name
                    try:
                        extracted_transcript = transcribe_indian_english_audio(tmp_path)
                    except Exception as err:
                        st.error(f"Whisper transcription error: {err}")
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                
                if extracted_transcript:
                    st.success("Audio transcribed successfully.")

        elif mode == "Video Response Mode":
            st.subheader("Step 1: Record & Download Response Video")
            render_video_recorder_component()
            
            st.subheader("Step 2: Upload Saved Video File")
            uploaded_video = st.file_uploader("Upload video file (.webm, .mp4):", type=["webm", "mp4"])
            
            if uploaded_video is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                saved_video_filename = f"video_{ts}.webm"
                saved_video_path = os.path.join(VIDEO_STORAGE_DIR, saved_video_filename)
                
                with open(saved_video_path, "wb") as f:
                    f.write(uploaded_video.read())
                
                st.video(saved_video_path)
                st.success(f"Video saved as `{saved_video_filename}`")
                
                with st.spinner("Transcribing video audio track..."):
                    try:
                        extracted_transcript = transcribe_indian_english_audio(saved_video_path)
                        st.success("Video audio track transcribed successfully.")
                    except Exception as err:
                        st.error(f"Transcription error: {err}")

        # Verified Transcript Area
        final_response_text = st.text_area(
            "Verified Response Transcript (Auto-generated or edit directly):", 
            value=extracted_transcript, 
            height=140
        )

        if st.button("Submit Answer for Panel Evaluation"):
            if not final_response_text.strip():
                st.error("Please record an audio/video response or provide transcript text first.")
            else:
                with st.spinner("Evaluating response against panel rubrics..."):
                    c_name = st.session_state['candidate_name']
                    eval_prompt = f"""
                    Act as strict placement auditor at IPER Bhopal.
                    Evaluate response for candidate: {c_name}.
                    IMPORTANT: Address {c_name} by name inside each feedback section.

                    Question: {st.session_state['current_question']}
                    Candidate Response: {final_response_text}
                    Resume Context: {json.dumps(st.session_state.get('resume_details', {}))}
                    Mode: {mode}

                    Return response in VALID JSON strictly matching this structure:
                    {{
                        "CandidateName": "{c_name}",
                        "GradingScore": <0-100 integer>,
                        "ExecutiveSummary": "<Greeting addressing {c_name} with overall performance summary>",
                        "TechnicalAssessment": "<Evaluation of domain accuracy, directly addressing {c_name}>",
                        "CommunicationAssessment": "<Assessment of structure, flow, and delivery, addressing {c_name}>",
                        "ResumeAlignment": "<How well {c_name} leveraged their resume background>",
                        "KeyFlaws": "<Specific technical or structural errors>",
                        "CorrectiveSteps": "<Actionable steps for {c_name} to improve>",
                        "Benchmark100Answer": "<Comprehensive model answer>"
                    }}
                    """
                    raw_eval = get_groq_response(eval_prompt)
                    
                    try:
                        clean_json = raw_eval.replace("```json", "").replace("```", "").strip()
                        eval_data = json.loads(clean_json)
                        
                        st.subheader(f"Panel Evaluation Report for {c_name}")
                        score = eval_data['GradingScore']
                        
                        if score < 50:
                            st.error(f"Score: {score}/100 - Below Placement Standard")
                        elif score < 75:
                            st.warning(f"Score: {score}/100 - Average (Needs Refinement)")
                        else:
                            st.success(f"Score: {score}/100 - High Performance")

                        if "ExecutiveSummary" in eval_data:
                            st.info(eval_data["ExecutiveSummary"])

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("### Technical Assessment")
                            st.write(eval_data['TechnicalAssessment'])
                            st.markdown("### Communication & Delivery")
                            st.write(eval_data['CommunicationAssessment'])
                        with c2:
                            st.markdown("### Resume Alignment")
                            st.write(eval_data['ResumeAlignment'])
                            st.markdown("### Key Flaws Identified")
                            st.write(eval_data['KeyFlaws'])

                        st.markdown(f"### Mentorship & Corrective Steps for {c_name}")
                        st.write(eval_data['CorrectiveSteps'])

                        with st.expander("View Benchmark 100/100 Model Answer"):
                            st.write(eval_data['Benchmark100Answer'])

                        st.session_state["history"].append({
                            "Timestamp": time.strftime("%Y-%m-%d %H:%M"),
                            "Candidate": c_name,
                            "Mode": mode,
                            "Domain": f"{category}" + (f" - {selected_spec}" if selected_spec else ""),
                            "Question": st.session_state['current_question'],
                            "Score": score,
                            "VideoFile": saved_video_filename
                        })
                        st.success("Attempt logged in sidebar history.")

                    except Exception:
                        st.markdown(raw_eval)

# TAB 3: DASHBOARD TAB
with tab_dashboard:
    st.header("Performance Dashboard")
    if not st.session_state["history"]:
        st.info("No practice attempts logged yet. Complete a session in the Practice Tab.")
    else:
        df = pd.DataFrame(st.session_state["history"])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Practice Attempts", len(df))
        m2.metric("Mean Assessment Score", f"{df['Score'].mean():.1f} / 100")
        m3.metric("Peak Score Recorded", f"{df['Score'].max()} / 100")

        st.markdown("---")
        st.subheader("Score Progress")
        st.line_chart(df, y="Score")

        st.subheader("Historical Log Table")
        st.dataframe(df, use_container_width=True)
