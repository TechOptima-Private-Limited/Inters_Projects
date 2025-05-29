 📑 Multi-Resume Parser & Summarizer App (with Local LLM - OptGPT)

This Streamlit-based web app helps recruiters, HR professionals, and hiring managers upload and parse multiple resumes (PDFs), extract important details, generate AI-powered summaries using a local LLM (OptGPT), and evaluate candidates using a simple match-scoring system. MongoDB is used to store parsed resumes and logs, with fallback to dummy data when MongoDB is unavailable.

---

 🚀 Features

- 📤 Upload up to 5 resumes at a time (PDF format)
- 🧠 Generate AI summaries of each resume using OptGPT (via Ollama)
- 🧮 Automatically calculate match percentage based on:
  - Experience
  - Education
  - Technical & Soft skills
  - Location
- 🧾 View recent resume records and query insights
- 🧰 Admin control to reset the database
- 🗨️ Ask questions like:
  - "Which candidates have strong soft skills?"
  - "Who has more than 5 years of experience?"
  - "Which resumes have the highest match percentage?"

---

🛠 Tech Stack

Tool/Library     | Purpose                                 

| [Streamlit](https://streamlit.io/)        | Web UI                              
| [PyMongo](https://pymongo.readthedocs.io/) | MongoDB integration                  
| [Ollama](https://ollama.com) + OptGPT     | Local LLM for AI-generated summaries 
| `extract_resume_data()` in `parserb.py`   | Resume PDF parsing                   
| [requests](https://docs.python-requests.org/) | Communicating with local model server 


 📂 Folder Structure

.
├── app.py                    # Main Streamlit app
├── parserb.py               # Resume data extractor logic
├── uploads/                 # Uploaded resume PDFs
├── requirements.txt         # Dependencies
└── README.md                # This file
