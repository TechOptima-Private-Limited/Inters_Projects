 👨‍💼 DrLoE: Life of an Employee – AI-Powered Attrition Insights

DrLoE is a Streamlit-based interactive tool that combines traditional HR analytics with OptGPT, a local large language model, to provide real-time predictions and explainable AI insights on employee attrition.

It is designed to support HR professionals, business analysts, and organizational leaders in understanding, predicting, and managing employee churn with both data and reasoning.

---

🚀 Features

- 📊 Upload your own employee dataset or explore with default sample data
- 👤 Online prediction: Predict attrition risk for a single employee using a simple form
- 📁 Batch prediction: Upload a CSV file of multiple employees to get risk assessments for all
- 🧠 Ask open-ended, natural language questions like:
  - “Why might employees leave the company?”
  - “Which department has the highest churn?”
- 💡 Explainable AI: Get reasoned responses powered by OptGPT for transparency

 🧠 How It Works

DrLoE is backed by a local LLM (OptGPT via Ollama) that interprets employee features such as job satisfaction, overtime, department, salary, etc., and offers both predictive feedback and natural language analysis.

Powered by:
- 📍 Streamlit — web interface
- 🧠 OptGPT (local LLM via [Ollama](https://ollama.com))
- 📊 Built-in logic or ML models (optional fallback)
- 🧮 Pandas and basic analytics for visualizations and filtering


📋 How to Use DrLoE

 Step 1: 🧠 Setup Local LLM with Ollama

Install Ollama and pull the OptGPT model:

ollama pull optgpt:7b
ollama run optgpt:7b

pip install -r requirements.txt
streamlit
pandas
requests
matplotlib
streamlit run app.py

