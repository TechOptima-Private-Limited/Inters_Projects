import streamlit as st
import requests
from pymongo import MongoClient, errors
import datetime
from parserb import extract_resume_data
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "optgpt:7b"
UPLOAD_FOLDER = "uploads"

try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    client.server_info()
    db = client["resume_parser_db"]
    logs_collection = db["logs"]
    resumes_collection = db["resumes"]
    mongo_available = True
except errors.ServerSelectionTimeoutError:
    mongo_available = False

DUMMY_RESUMES = [
    {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "technical_skills": ["python", "sql", "html"],
        "soft_skills": ["communication", "teamwork"],
        "experience": 4,
        "education": "Bachelor",
        "location": "New York",
        "timestamp": datetime.datetime.now(),
        "match_percentage": 85,
        "matched": True
    },
    {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "technical_skills": ["java", "c++"],
        "soft_skills": ["leadership"],
        "experience": 6,
        "education": "Master",
        "location": "San Francisco",
        "timestamp": datetime.datetime.now(),
        "match_percentage": 60,
        "matched": False
    }
]

def save_to_logs(question, answer):
    if mongo_available:
        logs_collection.insert_one({
            "question": question,
            "answer": answer,
            "timestamp": datetime.datetime.now()
        })

def save_resume_to_db(data):
    if mongo_available:
        data['timestamp'] = datetime.datetime.now()
        resumes_collection.insert_one(data)

def get_recent_resumes():
    if mongo_available:
        return list(resumes_collection.find().sort("timestamp", -1).limit(10))
    return DUMMY_RESUMES

def reset_database():
    if mongo_available:
        logs_collection.delete_many({})
        resumes_collection.delete_many({})

def query_model(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False})
        response.raise_for_status()
        return response.json().get("response", "No response from model.")
    except Exception as e:
        return f"Error querying model: {e}"

st.set_page_config(page_title="Resume Parser", layout="wide")
st.title("📑 Multi-Resume Parser & Summarizer")

left, middle, right = st.columns([1, 3, 2])

with left:
    st.markdown("## 📤 Upload Resumes")
    st.write("Upload up to 5 resumes in PDF format. The app will extract key data and generate  summaries.")

with middle:
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        for uploaded_file in uploaded_files[:5]:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            data = extract_resume_data(file_path)

            name = data.get("name", "N/A")
            email = data.get("email", "N/A")
            tech_skills = data.get("technical_skills", [])
            soft_skills = data.get("soft_skills", [])
            years_exp = data.get("experience", 0)
            education = data.get("education", "None")
            location = data.get("location", "")

            match_score = 0
            if years_exp >= 3:
                match_score += 30
            if education.lower() in ["bachelor", "master", "phd"]:
                match_score += 30
            if len(tech_skills) >= 2:
                match_score += 20
            if len(soft_skills) >= 1:
                match_score += 10
            if location:
                match_score += 10

            matched = match_score >= 70

            resume_record = {
                "name": name,
                "email": email,
                "technical_skills": tech_skills,
                "soft_skills": soft_skills,
                "experience": years_exp,
                "education": education,
                "location": location,
                "match_percentage": match_score,
                "matched": matched
            }

            save_resume_to_db(resume_record)

            st.success(f"✅ Parsed Resume: {name}")
            st.markdown(f"*Email:* {email}")
            st.markdown(f"*Skills:* {', '.join(tech_skills)}")
            st.markdown(f"*Experience:* {years_exp} years")
            st.markdown(f"*Education:* {education}")
            st.markdown(f"*Location:* {location}")
            st.markdown(f"*Match Percentage:* {match_score}%")

            with st.spinner("🧠 Generating OPGPT Summary..."):
                summary_prompt = (
                    f"Summarize this candidate:\n"
                    f"Name: {name}\nEmail: {email}\nExperience: {years_exp} years\n"
                    f"Education: {education}\nLocation: {location}\n"
                    f"Technical Skills: {', '.join(tech_skills)}\n"
                    f"Soft Skills: {', '.join(soft_skills)}"
                )
                ai_summary = query_model(summary_prompt)

            st.markdown("### 🧠 OPGPT Summary")
            st.write(ai_summary)
            save_to_logs(f"Summary for {name}", ai_summary)

with right:
    st.markdown("## ℹ️ About")
    st.write(
        "**This tool helps recruiters and hiring managers analyze multiple resumes " \
        "using AI-generated summaries and basic matching logic based on skills, education, experience, and more.**"
    )

    st.markdown("### How to Use This Tool:")
    st.write(
        """
        1. Upload up to 5 candidate resumes in PDF format.
        2. Each resume will be parsed to extract name, skills, education, etc.
        3. You will see the match percentage and AI-generated summary for each resume.
        4. Use the sidebar to review recent resumes and logs.
        """
    )

    st.markdown("### Sample Questions You Can Explore:")
    st.write(
        """
        - Which candidates have strong technical and soft skills?
        - How much experience do most applicants have?
        - What is the average match percentage of resumes?
        - Which candidates are suitable based on education level?
        """
    )

    st.markdown("### Target Users:")
    st.write(
        """
        - **Recruiters & HR Professionals**
        - **Hiring Managers**
        - **Startups & Small Businesses**
        - **AI and HR Tech Enthusiasts**
        """
    )

with st.sidebar:
    st.image("tech optima.png", width=250)
    st.markdown("## 🗂 Recent Resumes")
    recent_resumes = get_recent_resumes()
    if not mongo_available:
        st.warning("⚠️ Showing dummy resumes because MongoDB is unavailable.")

    if recent_resumes:
        for res in recent_resumes:
            st.markdown(f"**{res.get('name', 'N/A')}**")
            st.markdown(f"- Match: {res.get('match_percentage', 0)}%")
            st.markdown(f"- Experience: {res.get('experience', 'N/A')} years")
            st.markdown(f"- Education: {res.get('education', 'N/A')}")
            st.markdown("---")
    else:
        st.write("No resumes found.")

    st.markdown("## ⚠️ Admin Controls")
    if st.button("🧹 Reset Database"):
        reset_database()
        st.success("Database has been reset.")

st.header("💬 Ask Resume Insights OptGPT ")

user_prompt = st.text_area("Ask something about the uploaded resumes:")

def query_ollama_model_with_data(prompt, resumes):
    context = "\n".join([
        f"Name: {r['name']}, Email: {r['email']}, Skills: {', '.join(r.get('technical_skills', []))}, "
        f"Soft Skills: {', '.join(r.get('soft_skills', []))}, Experience: {r.get('experience')} years, "
        f"Education: {r.get('education')}, Location: {r.get('location')}, Match: {r.get('match_percentage')}%"
        for r in resumes
    ])

    full_prompt = f"Here is the candidate data:\n{context}\n\nAnswer this: {prompt}"
    return query_model(full_prompt)

if st.button("🧠 Get OPGPT Insight"):
    resumes_data = get_recent_resumes() if mongo_available else DUMMY_RESUMES

    if not resumes_data:
        st.warning("No resume data found.")
    elif not user_prompt.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("OptGPT is analyzing..."):
            ai_response = query_ollama_model_with_data(user_prompt, resumes_data)
        st.markdown("### OPGPT Response")
        st.write(ai_response)
        save_to_logs(user_prompt, ai_response)

