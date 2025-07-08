# 🏥 Secure Healthcare Chatbot with Streamlit, ollama, Groq, and Guardrails
 
This project is a **Streamlit-based AI chatbot** focused on **healthcare-related** queries. It integrates **Groq's LLM API**, **Ollama local models**, and **Guardrails AI** to ensure safety through **profanity filtering**, **toxicity checks**, and **custom LLM output validation**.
 
---
 
## 🔐 Features
 
- ✅ **Healthcare-Only Chatbot** — Rejects non-healthcare topics (e.g., sports, jokes, politics)
- ⚠️ **Profanity & Toxicity Filters** using Guardrails' built-in validators
- 🧠 **Custom LLM Output Validator** — Detects unsafe advice like medication dosages
- 🔄 **Support for Groq API** (`llama3`, `deepseek`) and **Ollama local models**
- 🛠️ Streamlit UI with selectable model and guardrail settings
- 🧪 Built-in test prompts for safety testing and adversarial evaluation
 
---
 
## 🧱 Tech Stack
 
- **Frontend/UI:** Streamlit
- **LLM Backend:** Groq API, Ollama (local)
- **Safety/Validation:** Guardrails AI (`ProfanityFree`, `ToxicLanguage`)
- **Environment Management:** dotenv
- **PDF Warning Suppression:** `pypdfium2` (if used in related components)
 
---
 
 
## 🚀 Getting Started
 
1. Clone the Repository
```bash
# git clone repository link
# cd foldername
```
 
2. Install Dependencies
```bash
pip install -r requirements.txt
```
 
3. Setup Environment Variables
Create a .env file in the root directory with the following:
```bash

OLLAMA_URL=optgpt_url
```
 
4. Run the App
```bash
streamlit run app.py
```
 
 
 