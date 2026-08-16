import os
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
# 1. STREAMLIT CONFIG & SYSTEM PATHS
# ------------------------------------------------------------------------------
st.set_page_config(page_title="IPER Placement & ATS Portal", layout="wide")

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" + os.pathsep + "/usr/local/bin"

VIDEO_STORAGE_DIR = "saved_videos"
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. BULLETPROOF STYLING (PREVENTS MATERIAL ICON BLEED & DARK OVERLAY BOXES)
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

        /* Prevent breaking Material Icons (e.g. arrow_drop_down) */
        p, h1, h2, h3, h4, h5, h6, label, span, li, td, th {
            color: #0F172A !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #0F172A !important;
        }

        /* Expander Styling */
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

        /* File Uploaders */
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

        /* Action Buttons */
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

        /* Sidebar Navigation Radio Buttons */
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            margin-bottom: 8px !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            border-color: #2563EB !important;
            background-color: #EFF6FF !important;
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
You are a senior MBA Placement Panel Auditor and HR ATS Evaluator at IPER Bhopal.
Address the candidate respectfully by name in all responses.
Provide exact, detailed, professional analysis.
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
# 5. SIDEBAR NAVIGATION CONTROLS (RESUME TOP, INTERVIEW BELOW)
# ------------------------------------------------------------------------------
st.sidebar.title("IPER MOCK AI")
st.sidebar.markdown("### Navigation Menu")

# Primary navigation panel
selected_nav = st.sidebar.radio(
    "Select Portal Section:",
    [
        "📄 Resume ATS Analyzer", 
        "🎙️ Practice Interview Terminal", 
        "📚 Interview Preparation Module", 
        "📊 Dashboard & Analytics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Attempt Logs")
if not st.session_state["history"]:
    st.sidebar.info("No practice attempts recorded yet.")
else:
    for idx, session in enumerate(reversed(st.session_state["history"]), 1):
        with st.sidebar.expander(f"Attempt #{len(st.session_state['history']) - idx + 1}: {session['Timestamp']}"):
            st.markdown(f"**Candidate:** {session.get('Candidate', 'N/A')}")
            st.markdown(f"**Domain:** {session['Domain']}")
            st.markdown(f"**Score:** {session['Score']}/100")

# ------------------------------------------------------------------------------
# 6. PORTAL SECTION ROUTING
# ------------------------------------------------------------------------------

# ==============================================================================
# SECTION 1: RESUME ATS ANALYZER (TOP ITEM IN SIDEBAR)
# ==============================================================================
if selected_nav == "📄 Resume ATS Analyzer":
    st.title("IPER Resume ATS Analyzer")
    st.caption("Upload candidate resume and target Job Description (JD) for ATS scoring, structural breakdown, and edits.")

    col_res, col_jd = st.columns(2)

    with col_res:
        st.subheader("1. Candidate Resume Upload")
        ats_resume_file = st.file_uploader("Upload Resume (PDF, Word DOCX):", type=["pdf", "docx", "doc"], key="ats_res_upload")
        
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
                
                st.session_state["candidate_name"] = c_name
                st.session_state["resume_details"] = resume_data
                st.success(f"Resume profile extracted for **{st.session_state['candidate_name']}**")

    with col_jd:
        st.subheader("2. Target Job Description (JD)")
        jd_input_option = st.radio("Provide JD via:", ["Upload File (PDF/DOCX/Image)", "Paste Text Direct"], horizontal=True)
        
        jd_text = ""
        if jd_input_option == "Upload File (PDF/DOCX/Image)":
            jd_file = st.file_uploader("Upload Job Description:", type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], key="ats_jd_upload")
            if jd_file:
                jd_text = extract_text_from_file(jd_file)
                st.info(f"Loaded JD File: `{jd_file.name}`")
        else:
            jd_text = st.text_area("Paste Job Description (JD) Text Here:", height=180, placeholder="Paste requirements, job roles, and qualifications...")

    st.markdown("---")

    if st.button("Run Comprehensive ATS Audit & Alignment Analysis", use_container_width=True):
        if not ats_resume_file:
            st.error("Please upload candidate resume first.")
        elif not jd_text.strip():
            st.error("Please upload or paste a Job Description (JD).")
        else:
            with st.spinner("Executing ATS match evaluation..."):
                candidate_name = st.session_state.get("candidate_name", "Candidate")
                resume_content = json.dumps(st.session_state.get("resume_details", {}))
                
                ats_prompt = f"""
                Act as an elite Corporate ATS Auditor & Placement Director at IPER Bhopal.
                Perform an ATS match evaluation for candidate '{candidate_name}'.

                Candidate Resume Context:
                {resume_content}

                Target Job Description (JD):
                {jd_text}

                Return ONLY valid JSON with this exact key structure:
                {{
                    "CandidateName": "{candidate_name}",
                    "ATSScore": <0-100 integer score>,
                    "Category": "<MUST be exactly one of: Excellent | Good | Average>",
                    "ExecutiveSummary": "<Greeting addressing {candidate_name} by name with summary of fit>",
                    "Strengths": ["<Strength 1>", "<Strength 2>", "<Strength 3>"],
                    "AreasOfImprovement": ["<Area 1>", "<Area 2>", "<Area 3>"],
                    "RecommendedChanges": ["<Specific bullet edit 1>", "<Keyword addition 2>", "<Formatting tip 3>"],
                    "MissingKeywords": ["<Keyword 1>", "<Keyword 2>", "<Keyword 3>"]
                }}
                """
                raw_ats_response = get_groq_response(ats_prompt)
                
                try:
                    clean_ats = raw_ats_response.replace("```json", "").replace("```", "").strip()
                    ats_result = json.loads(clean_ats)
                    
                    st.subheader(f"ATS Evaluation Audit for {candidate_name}")
                    
                    score = ats_result.get("ATSScore", 70)
                    category_rating = ats_result.get("Category", "Good")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Overall ATS Compatibility Score", f"{score} / 100")
                    with m2:
                        if category_rating == "Excellent":
                            st.success(f"Resume Strength Rating: **EXCELLENT**")
                        elif category_rating == "Good":
                            st.info(f"Resume Strength Rating: **GOOD**")
                        else:
                            st.warning(f"Resume Strength Rating: **AVERAGE**")
                    
                    st.info(ats_result.get("ExecutiveSummary", ""))
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### Strengths Highlighted")
                        for str_item in ats_result.get("Strengths", []):
                            st.markdown(f"- **{str_item}**")
                            
                        st.markdown("### Missing Key Terms")
                        for kw in ats_result.get("MissingKeywords", []):
                            st.markdown(f"- `{kw}`")
                            
                    with c2:
                        st.markdown("### Areas of Improvement")
                        for imp in ats_result.get("AreasOfImprovement", []):
                            st.markdown(f"- {imp}")
                            
                        st.markdown("### Recommended Actionable Revisions")
                        for chg in ats_result.get("RecommendedChanges", []):
                            st.markdown(f"- {chg}")

                except Exception:
                    st.markdown(raw_ats_response)

# ==============================================================================
# SECTION 2: MOCK INTERVIEW TERMINAL (POSITIONED BELOW RESUME SECTION)
# ==============================================================================
elif selected_nav == "🎙️ Practice Interview Terminal":
    st.title("IPER Practice & Interview Evaluation Terminal")
    st.caption("Conduct audio or video mock interviews with personalized AI feedback.")
    
    if st.session_state.get("resume_details"):
        st.success(f"Active Candidate Profile: **{st.session_state['candidate_name']}** (Loaded from Resume ATS section)")
    else:
        st.info("Tip: Upload candidate resume in the **Resume ATS Analyzer** sidebar section to activate tailored questions.")

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

    if st.button("Generate Interview Question", use_container_width=True):
        if category == "Resume-Based (Tailored)":
            if not st.session_state.get("resume_details"):
                st.warning("Please upload candidate resume first in the Resume ATS Analyzer sidebar section.")
            else:
                prompt = f"""
                Act as a strict MBA interviewer at IPER Bhopal. 
                Generate ONE interview question tailored to {st.session_state['candidate_name']}'s resume profile:
                {json.dumps(st.session_state['resume_details'])}
                Difficulty: {difficulty}
                Return ONLY the question text directly.
                """
                with st.spinner(f"Generating question for {st.session_state['candidate_name']}..."):
                    st.session_state["current_question"] = get_groq_response(prompt)
        else:
            ctx = f"Domain: {category}, Difficulty: {difficulty}"
            if selected_spec: ctx += f", Specialization: {selected_spec}"
            if selected_comp: ctx += f", Target Company: {selected_comp}"
            if st.session_state["resume_details"]: ctx += f", Resume Context: {json.dumps(st.session_state['resume_details'])}"
            
            prompt = f"Act as an MBA interviewer at IPER Bhopal. Generate ONE question for '{st.session_state['candidate_name']}' based on: {ctx}. Return ONLY question text."
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
            st.subheader("Step 1: Record & Save Video Response")
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

        final_response_text = st.text_area(
            "Verified Response Transcript:", 
            value=extracted_transcript, 
            height=140
        )

        if st.button("Submit Answer for Panel Evaluation"):
            if not final_response_text.strip():
                st.error("Please record an audio/video response or provide transcript text first.")
            else:
                with st.spinner("Evaluating response..."):
                    c_name = st.session_state['candidate_name']
                    eval_prompt = f"""
                    Act as strict placement auditor at IPER Bhopal.
                    Evaluate response for candidate: {c_name}.
                    Address {c_name} by name inside each feedback section.

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
                        "ResumeAlignment": "<How well {c_name} leveraged their background>",
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

# ==============================================================================
# SECTION 3: INTERVIEW PREPARATION MODULE
# ==============================================================================
elif selected_nav == "📚 Interview Preparation Module":
    st.title("Interview Preparation Module")
    st.caption("Select a domain and question to study the Objective, Answering Structure, and Benchmark Sample Answer.")
    
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

# ==============================================================================
# SECTION 4: DASHBOARD & ANALYTICS
# ==============================================================================
elif selected_nav == "📊 Dashboard & Analytics":
    st.title("Performance Dashboard & Analytics")
    if not st.session_state["history"]:
        st.info("No practice attempts logged yet. Complete a session in the Practice Interview Terminal.")
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
