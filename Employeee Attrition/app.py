import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
import json

# --- Constants ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "optgpt:7b"

# ========== Query OptGPT with Data ==========
def query_ollama_model_with_data(prompt, df):
    data_sample = df.head(5).to_dict(orient='records')
    column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
    general_info = f"""
You are a data expert AI assistant. The dataset you should refer to is about employee attrition. Answer only based on this dataset, do not use external knowledge.

### Dataset Overview:
- Shape: {df.shape}
- Columns and Types:
{column_info}

### Sample Records (first 5 rows):
{json.dumps(data_sample, indent=2)}

### Question:
{prompt}

Give your answer based only on the data shown above. If data is insufficient to answer precisely, state that.
    """

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": general_info,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            st.error(f"Ollama Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error contacting Ollama: {e}")
        return None

# ========== Load or Upload Datasets ==========
@st.cache_data
def load_default_data():
    left = pd.read_csv("employee_who_left.csv")
    existing = pd.read_csv("existing_employee.csv")
    left['Churn'] = 1
    existing['Churn'] = 0
    data = pd.concat([left, existing], ignore_index=True)
    return data

# ========== Main App ==========
def main():
    st.set_page_config(page_title="DrLoE - Employee Attrition via OptGPT", layout="wide")

    st.sidebar.image("tech optima.png", use_container_width=True)
    st.sidebar.subheader("\U0001F4C1 Upload Data (optional)")
    new_left_file = st.sidebar.file_uploader("employee_who_left.csv", type="csv")
    new_existing_file = st.sidebar.file_uploader("existing_employee.csv", type="csv")
    batch_file = st.sidebar.file_uploader("\U0001F4E4 Upload Batch Employee Data for Prediction", type="csv")

    if new_left_file and new_existing_file:
        left = pd.read_csv(new_left_file)
        existing = pd.read_csv(new_existing_file)
        left['Churn'] = 1
        existing['Churn'] = 0
        df = pd.concat([left, existing], ignore_index=True)
    else:
        df = load_default_data()

    main_col, about_col = st.columns([3, 1])

    with main_col:
        st.title("\U0001F9D0 DrLoE - Employee Attrition Analysis via OptGPT")

        if st.button("About Author"):
            st.write("""
            \U0001F469‍\U0001F4BB 
            \U0001F50D DrLoE helps understand why employees leave and offers AI-driven insights using OptGPT.
            """)

        st.header("\U0001F4AC Ask DrLoE OptGPT")
        user_prompt = st.text_area("Type a question to the model...")

        if st.button("Generate OPGPT Insight"):
            if user_prompt.strip():
                with st.spinner("OptGPT is thinking..."):
                    ai_response = query_ollama_model_with_data(user_prompt, df)
                    if ai_response:
                        st.subheader("\U0001F916 OPGPT's Response:")
                        st.write(ai_response)
                    else:
                        st.warning("No response received.")



    st.subheader("\U0001F4CA Sample of the Data")
    st.dataframe(df.sample(10))

    st.header("\U0001F4C8 EDA: Departments of Employees Who Left")
    fig1, ax = plt.subplots()
    df[df['Churn'] == 1]['dept'].value_counts().plot(kind='bar', ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Employees who left by department")
    st.pyplot(fig1)

    st.header("\U0001F50D Predict Attrition for a New Employee")
    with st.form("predict_form"):
        satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        last_eval = st.slider("Last Evaluation", 0.0, 1.0, 0.6)
        num_projects = st.number_input("Number of Projects", 1, 10, 3)
        avg_monthly_hours = st.number_input("Average Monthly Hours", 80, 400, 160)
        time_spend = st.slider("Time Spent at Company (Years)", 0, 10, 3)
        work_accident = st.selectbox("Work Accident", [0, 1])
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
        dept = st.selectbox("Department", df['dept'].unique())
        salary = st.selectbox("Salary Level", df['salary'].unique())
        submitted = st.form_submit_button("Predict via OptGPT")

    if submitted:
        employee_input = {
            "satisfaction_level": satisfaction,
            "last_evaluation": last_eval,
            "number_project": num_projects,
            "average_montly_hours": avg_monthly_hours,
            "time_spend_company": time_spend,
            "Work_accident": work_accident,
            "promotion_last_5years": promotion_last_5years,
            "dept": dept,
            "salary": salary
        }

        prompt = f"""
Based on the dataset sample and this new employee's profile, predict whether the employee is likely to leave the company or not (Churn = 1 or 0), and briefly explain your reasoning.

Employee Details:
{json.dumps(employee_input, indent=2)}
        """

        with st.spinner("OptGPT is analyzing..."):
            result = query_ollama_model_with_data(prompt, df)
            if result:
                st.subheader("\U0001F9E0 OptGPT Prediction:")
                st.write(result)
            else:
                st.error("Prediction failed from OptGPT.")

    st.header("\U0001F4E6 Batch Prediction for Multiple Employees")
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.write("✅ Batch file uploaded successfully:")
        st.dataframe(batch_df)

        results = []
        churn_flags = []
        with st.spinner("OptGPT is predicting for all employees..."):
            for i, row in batch_df.iterrows():
                emp_profile = row.to_dict()
                prompt = f"""
Based on the dataset sample and this employee's profile, predict whether the employee is likely to leave the company or not (Churn = 1 or 0), and explain your reasoning.

Employee Details:
{json.dumps(emp_profile, indent=2)}
                """
                result = query_ollama_model_with_data(prompt, df)
                if result:
                    prediction = "1" if "churn = 1" in result.lower() or "likely to leave" in result.lower() else "0"
                else:
                    prediction = "Error"

                churn_flags.append(prediction)
                results.append({
                    "Employee #": i + 1,
                    "Prediction": prediction,
                    "Explanation": result if result else "Failed"
                })

        result_df = pd.DataFrame(results)
        result_df["Churn_Flag"] = churn_flags

        st.subheader("\U0001F4CB Batch Prediction Results:")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("\U0001F4E5 Download Results CSV", csv, "batch_predictions.csv", "text/csv")

        churn_count = result_df[result_df["Churn_Flag"] == "1"].shape[0]
        safe_count = result_df[result_df["Churn_Flag"] == "0"].shape[0]
        st.success(f"\U0001F50D Out of {len(result_df)} employees, {churn_count} are predicted to leave, {safe_count} to stay.")

        st.subheader("\U0001F4CA Churn Distribution")
        fig2, ax2 = plt.subplots()
        ax2.pie([churn_count, safe_count], labels=["Will Leave", "Will Stay"], autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)

    with about_col:
        st.header("ℹ️ About ")
        st.markdown("""
                     ### About DrLoE:
        DrLoE (Life of an Employee) leverages OptGPT, a powerful AI language model, to analyze employee data and provide actionable insights on attrition. It combines data-driven predictions with explainable AI reasoning to assist better decision-making.

        ### How to Use the Tool:
        - Upload employee datasets on the sidebar or use the default sample data.
        - View sample records and explore employee attrition trends.
        - Use the form to predict attrition risk for a single employee (online prediction).
        - Upload batch employee data CSV to predict attrition for multiple employees at once.
        - Ask open-ended questions to the AI model about attrition insights.

        ### Sample Questions:
        - Why might employees leave the company?
        - Which departments have the highest employee churn?
        - What factors contribute most to attrition?

        ### Target Users:
        - HR managers and recruiters.
        - Business analysts and data scientists.
        - Organizational leaders interested in workforce insights. """)

if __name__ == "__main__":
    main()
