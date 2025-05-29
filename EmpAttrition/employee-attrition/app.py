import streamlit as st
import requests
 
# Local LLM Config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "optgpt:7b"
 
# Page Setup
st.set_page_config(layout="wide", page_title='DrLoE - Employee Attrition via OptGPT')
 
# Define page columns
left_col, middle_col, right_col = st.columns([1.3, 2.5, 1.2])
 
# LEFT SIDEBAR-LIKE SECTION
with left_col:
    st.image("image.png", width=200)  # Make sure image.png is in same directory as app.py
 
    st.markdown("### 📂 Upload Data (optional)")
    st.file_uploader("employee_who_left.csv", type=["csv"], key="file1")
    st.file_uploader("existing_employee.csv", type=["csv"], key="file2")
    st.file_uploader("📄 Upload Batch Employee Data for Prediction", type=["csv"], key="file3")
 
# MIDDLE SECTION - MAIN Q&A INTERFACE
with middle_col:
    st.markdown("## DrLoE - Employee Attrition Analysis via OptGPT")
   
    st.markdown("### 🧠 Ask DrLoE OptGPT")
    user_question = st.text_area("Type a question to the model...", placeholder="Which departments have the highest employee attrition?")
 
    if st.button("Generate OPGPT Insight"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            prompt = f"""
You are a workplace analytics expert. Analyze the employee attrition data and answer the question below:
 
Question: {user_question}
 
If needed, generate simple tabular summaries or insights.
"""
            with st.spinner("Generating response..."):
                try:
                    payload = {
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": False
                    }
                    response = requests.post(OLLAMA_URL, json=payload)
                    result = response.json().get("response", "No response from OptGPT.")
                except Exception as e:
                    result = f"❌ Error: {e}"
 
            st.markdown("### 🤖 OPGPT's Response:")
            st.write(result)
 
    st.markdown("---")
    st.markdown("### 📝 Online Prediction Form")
 
    name = st.text_input("Employee Name")
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
    projects = st.number_input("Number of Projects", 1, 10, 3)
    hours = st.number_input("Average Monthly Hours", 50, 400, 160)
    years = st.number_input("Years at Company", 0, 40, 3)
    accident = st.selectbox("Had Work Accident?", ["No", "Yes"])
    promoted = st.selectbox("Promoted in Last 5 Years?", ["No", "Yes"])
    dept = st.selectbox("Department", [
        "sales", "technical", "support", "IT", "hr", "accounting",
        "marketing", "product_mng", "randD", "management"
    ])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])
 
    if st.button("Submit and Predict with AI"):
        employee_data = {
            "Name": name,
            "Satisfaction": satisfaction,
            "Evaluation": evaluation,
            "Projects": projects,
            "Monthly Hours": hours,
            "Years at Company": years,
            "Work Accident": accident,
            "Promoted": promoted,
            "Department": dept,
            "Salary": salary
        }
 
        prompt = f"""
You are an expert HR analyst. Based on the following employee data, predict whether the employee is likely to leave and explain why.
 
Employee Data:
{employee_data}
 
Respond in one paragraph with a clear prediction and reason.
"""
 
        with st.spinner("Analyzing employee details with local AI model..."):
            try:
                payload = {
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                }
                response = requests.post(OLLAMA_URL, json=payload)
                result = response.json().get("response", "No response from model.")
            except Exception as e:
                result = f"❌ Error: {e}"
 
        st.markdown("### 🔍 Prediction Result:")
        st.write(result)
 
# RIGHT INFO PANEL
with right_col:
    st.markdown("### ℹ️ About")
    st.markdown("**About DrLoE:**\n\nDrLoE (Life of an Employee) leverages OptGPT, a powerful AI model, to analyze employee data and provide insights on attrition.")
 
    st.markdown("### 📘 How to Use the Tool")
    st.markdown("""- Upload employee datasets on the left.
- Ask open-ended questions to the AI in the center.
- Submit individual employee data for prediction.""")
 
    st.markdown("### 💬 Sample Questions")
    st.markdown("""- Why might employees leave the company?  
- Which departments have the highest employee attrition?  
- What factors contribute most to attrition?""")
 
    st.markdown("### 🎯 Target Users")
    st.markdown("""- HR Analysts  
- Business Leaders  
- Data Enthusiasts""")

 