# 🏥 Secure Healthcare Chatbot with OptGPT and Guardrails

This project is an AI-powered **healthcare chatbot** built with **Streamlit** and backed by the `optgpt:7b` model (served via Ollama). It is designed to answer **only healthcare-related questions** while **blocking inappropriate, toxic, or irrelevant queries** using **Guardrails AI**.

> ⚠️ The chatbot strictly refuses to respond to any non-healthcare, profane, or toxic prompts.

---

## 🚀 Features

- ✅ **Healthcare-focused LLM answers** using `optgpt:7b`
- 🔐 **Input and output validation** with Guardrails' `ProfanityFree` and `ToxicLanguage` validators
- 📄 **Context-aware** responses using a local healthcare PDF as reference
- 🧼 Cleans model reasoning (`<think>...</think>`) before displaying
- 🧠 Educates users with example questions
- 🧩 UI built with **Streamlit**, easy to run and customize

---

## 📁 Folder Structure

secure_chatbot/
├── app.py # Main Streamlit application
├── .env # Environment file with API settings (excluded from Git)
├── data/
│ ├── healthcare_policy.pdf # PDF used as context
│ ├── secure_medical_prompt_injection_policy.pdf
│ └── techoptima-01.jpg # Branding image in sidebar
├── README.md # Project overview
├── requirements.txt # Required Python packages
└── .gitignore # Ignores .env and cache files

yaml
Copy
Edit

---

## ⚙️ Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running the `optgpt:7b` model
- [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- `streamlit`, `dotenv`, `requests`, `re`

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/secure_chatbot.git
cd secure_chatbot

2. Create & Activate Virtual Environment (Optional but Recommended)
bash

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install Dependencies
bash

pip install -r requirements.txt
4. Add Environment Variables
Create a .env file in the root directory with your Ollama API settings:

env

OLLAMA_URL=http://localhost:11434/api/generate
Replace with your actual Ollama endpoint.

5. Run the App
bash

streamlit run app.py
Open your browser at: http://localhost:8501