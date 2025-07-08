# 🩺 Secure Healthcare Chatbot using OptGPT + Streamlit + Guardrails

This is a **secure, AI-powered healthcare chatbot** built with:
- **OptGPT (via Ollama)** as the LLM backend
- **Streamlit** for the web interface
- **Guardrails** for **profanity and toxicity filtering**
- **PDF knowledge ingestion** for contextual healthcare answers

> ⚠️ The chatbot **only answers healthcare-related queries**. It **blocks toxic, profane, or unrelated input** using AI safety guardrails.

---

## 📌 Features

- ✅ Answers **only healthcare-related** questions (e.g., symptoms, diagnosis, treatments, medical policies).
- ❌ Politely declines general knowledge, entertainment, or unrelated queries.
- 🔐 Uses **Guardrails AI** to prevent:
  - Profane input/output
  - Toxic or manipulative queries
- 🧼 Cleans `<think>` reasoning artifacts in OptGPT output
- 📄 Loads a **healthcare policy PDF** as reference material
- 🎯 Uses `.env` for secure API configuration

---

## 🛠️ Technologies Used

| Tool/Library      | Purpose                              |
|------------------|--------------------------------------|
| `OptGPT via Ollama` | AI model for answering queries       |
| `Streamlit`       | Frontend UI                          |
| `Guardrails AI`   | Validates safe input and output      |
| `dotenv`          | Loads API keys securely              |
| `PDF loader`      | Extracts text from a healthcare policy PDF |



## 🚀 How to Run the Chatbot Locally

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/TechOptima-Private-Limited/Inters_Projects.git
cd Inters_Projects/streamlit_chatbot



✅ 2. Set Up a Virtual Environment
bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


✅ 3. Install Dependencies
bash

pip install -r requirements.txt

Ensure guardrails, streamlit, requests, python-dotenv, and PyMuPDF are included in requirements.txt.

✅ 4. Add Your .env File
Create a .env file in the project root:
Edit
OLLAMA_URL=http://localhost:11434/api/generate
MODEL_NAME=optgpt:7b
❗ Make sure .env is listed in .gitignore to avoid leaking secrets.

✅ 5. Start the Streamlit App
bash
streamlit run app.py
This will launch the chatbot in your browser.

