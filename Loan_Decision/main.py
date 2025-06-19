


import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import *
from scipy.stats import norm, skew
import pandas_bokeh
import base64
import requests

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from po_db import (
    save_application,
    save_chat_log,
    save_exploration_log,
    save_analytics_log
)

pandas_bokeh.output_notebook()

OLLAMA_URL = "http://192.168.1.117:11434/api/generate"
MODEL_NAME = "optgpt:7b"

def query_optgpt(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response received from model.")
    except Exception as e:
        return f"Error querying OptGPT: {str(e)}"

def main():
    train = load_data()
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Data Exploration', 'Prediction', 'Analytics', 'OptGPT Chat'])

    if page == 'Homepage':
        st.title(':moneybag:Analytics and Machine Learning App')
        st.markdown('👈Select a page in the sidebar')
        st.markdown('This application performs machine learning predictions on loan applications and outputs predictions of approved or rejected applications.')

        st.markdown('This application provides:')
        st.markdown('● Machine learning prediction on loan applications.:computer:')
        st.markdown('● Data Exploration of the dataset used in training and prediction.:bar_chart:')
        st.markdown('● Custom data Visualization and Plots.:chart_with_upwards_trend:')

        if st.checkbox('Show raw Data'):
            st.dataframe(train)

    elif page == 'Data Exploration':
        st.title('Explore the Dataset')
        if st.checkbox('Show raw Data'):
            st.dataframe(train)

        st.markdown('### Analysing column distribution')
        all_columns_names = train.columns.tolist()
        selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)
        if st.button("Generate Plot"):
            st.success("Generating Customizable Bar Plot for {}".format(selected_columns_names))
            cust_data = train[selected_columns_names]
            st.bar_chart(cust_data)
            save_exploration_log("Bar chart generated", selected_columns_names)

        if st.checkbox("Show Shape"):
            st.write(train.shape)
            save_exploration_log("Viewed shape")

        if st.checkbox("Show Columns"):
            all_columns = train.columns.to_list()
            st.write(all_columns)
            save_exploration_log("Viewed column list")

        if st.checkbox("Summary"):
            st.write(train.describe())
            save_exploration_log("Viewed summary statistics")

        if st.checkbox("Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = train[selected_columns]
            st.dataframe(new_df)
            save_exploration_log("Viewed selected columns", selected_columns)

        if st.checkbox("Show Value Counts"):
            st.write(train.iloc[:, 0].value_counts())
            save_exploration_log("Viewed value counts of first column")

        if st.checkbox("Correlation Plot(Matplotlib)"):
            plt.matshow(train.corr())
            st.pyplot()
            save_exploration_log("Viewed correlation matrix (Matplotlib)")

        if st.checkbox("Correlation Plot(Seaborn)"):
            st.write(sns.heatmap(train.corr(), annot=True))
            st.pyplot()
            save_exploration_log("Viewed correlation matrix (Seaborn)")

    elif page == 'Analytics':
        st.title('Analytics.:bar_chart:')
        st.markdown('Upload your custom dataset and run visualisations.')
        uploaded_file = st.file_uploader("Upload a Dataset (CSV) for Analysis", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if st.checkbox('Show data'):
                st.dataframe(df)

            pd.set_option('plotting.backend', 'pandas_bokeh')
            st.subheader("Customizable Plot")
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "step", "point", "scatter", "barh", "map"])
            selected_columns_names = st.multiselect("Select Column(s) To Plot", all_columns_names)

            if st.button("Generate Plot"):
                st.success("Generating {} plot for {}".format(type_of_plot, selected_columns_names))
                save_analytics_log(uploaded_file.name, type_of_plot, selected_columns_names)

                if type_of_plot == 'scatter':
                    if len(selected_columns_names) == 1:
                        cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                        st.bokeh_chart(cust_plot)
                    else:
                        cust_plot = df[selected_columns_names].iloc[:, [0, 1]].plot.scatter(
                            x=selected_columns_names[0], y=selected_columns_names[1])
                        st.bokeh_chart(cust_plot)
                elif type_of_plot:
                    cust_plot2 = df[selected_columns_names].plot(kind=type_of_plot)
                    st.bokeh_chart(cust_plot2)

    elif page == 'OptGPT Chat':
        st.title('🔍 Ask OptGPT')
        user_prompt = st.text_area("Ask OptGPT a question related to loans, finance, or data insights:")
        if st.button("Send to OptGPT"):
            with st.spinner("💭 OptGPT is thinking..."):
                response = query_optgpt(user_prompt)
            save_chat_log(user_prompt, response)
            if "<think>" in response and "</think>" in response:
                thought_start = response.find("<think>") + len("<think>")
                thought_end = response.find("</think>")
                thoughts = response[thought_start:thought_end].strip()
                final_response = response[thought_end + len("</think>"):].strip()
            else:
                thoughts = None
                final_response = response
            st.markdown("**Model Response:**")
            st.write(final_response)
            if thoughts:
                with st.expander("🧠 OptGPT's Thought Process"):
                    st.markdown(thoughts)
                # st.markdown("**Model Response:**")
                # st.write(response)
                # save_chat_log(user_prompt, response)

    else:
        st.title('Modelling')
        model, _ = train_model(train)

        Current_Loan_Amount = st.number_input("Enter Loan Amount", 0, 90000000, 0)
        Credit_Score = st.number_input("Credit Score", 300, 1255, 300)
        Annual_Income = st.number_input("Annual Income", 0, 9000000, 0)
        Years_in_current_job = st.number_input("Enter Years in Current Job", 0, 20, 0)
        Term_Short_Term = st.selectbox("Loan Tenure", ['Short Term', 'Long Term'])
        Home_Ownership_Home_Mortgage = st.selectbox('Mortgage?', ["Yes", "No"])
        Home_Ownership_Own_Home = st.selectbox('Own Home?', ["Yes", "No"])
        Home_Ownership_Rent = st.selectbox('Rent?', ["Yes", "No"])

        uploaded_file = st.sidebar.file_uploader("Upload a CSV file for Batch Prediction", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.strip().str.replace(' ', '_')
            data2 = data.copy()
            data = pd.get_dummies(data, drop_first=True)
            for col in model.feature_names_in_:
                if col not in data.columns:
                    data[col] = 0
            data = data[model.feature_names_in_]
            prediction = model.predict(data)

            if st.sidebar.button("Prediction"):
                submit = data2
                submit['Loan_Status'] = prediction
                submit['Loan_Status'] = submit['Loan_Status'].map({1: 'Approved', 0: 'Rejected'})
                st.sidebar.info('Batch Prediction Completed!')
                def get_table_download_link(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="Predictions.csv">Download csv file</a>'
                    return href
                st.sidebar.markdown(get_table_download_link(submit), unsafe_allow_html=True)

        input_data = [[Current_Loan_Amount, Credit_Score, Annual_Income, Years_in_current_job,
                       1 if Term_Short_Term == 'Short Term' else 0,
                       1 if Home_Ownership_Home_Mortgage == 'Yes' else 0,
                       1 if Home_Ownership_Own_Home == 'Yes' else 0,
                       1 if Home_Ownership_Rent == 'Yes' else 0]]
        columns = ['Current_Loan_Amount', 'Credit_Score', 'Annual_Income', 'Years_in_current_job',
                   'Term_Short_Term', 'Home_Ownership_Home_Mortgage', 'Home_Ownership_Own_Home', 'Home_Ownership_Rent']
        P = pd.DataFrame(input_data, columns=columns)
        for col in model.feature_names_in_:
            if col not in P.columns:
                P[col] = 0
        P = P[model.feature_names_in_]
        prediction = model.predict(P)

        if st.button("Predict"):
            result = prediction
            decision = 'Approved' if result[0] else 'Rejected'
            save_application(P.to_dict(orient='records')[0], decision)
            st.success("Loan Application is {}".format(decision))

@st.cache_data
def train_model(train):
    le = LabelEncoder()
    cols = ['Term', 'Home Ownership']
    train['Loan Status'] = le.fit_transform(train['Loan Status'])
    train = pd.get_dummies(data=train, columns=cols, drop_first=True)
    X = train.drop(columns=['Purpose', 'Monthly Debt', 'Years of Credit History',
                            'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance',
                            'Maximum Open Credit', 'Bankruptcies', 'Tax Liens'])
    y = X['Loan Status'].values
    X = X.drop(columns=['Loan Status'])
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.2, random_state=42)
    model = ExtraTreesClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)

@st.cache_data
def load_data():
    train = pd.read_csv('credit_train.csv')
    train.drop_duplicates(keep=False, inplace=True)
    train['Years in current job'] = train['Years in current job'].replace({
        '10+ years': '10', '< 1 year': '1', '1 year': '1', '2 years': '2', '3 years': '3',
        '4 years': '4', '5 years': '5', '6 years': '6', '7 years': '7', '8 years': '8', '9 years': '9'
    })
    numeric_train = train.select_dtypes(include=['number'])
    Q1 = numeric_train.quantile(0.25)
    Q3 = numeric_train.quantile(0.75)
    IQR = Q3 - Q1
    train = train[~((numeric_train < (Q1 - 0.5 * IQR)) | (numeric_train > (Q3 + 1.5 * IQR))).any(axis=1)]
    train['Years in current job'] = pd.to_numeric(train['Years in current job'])
    train.drop_duplicates('Loan ID', inplace=True)
    train = train.drop(columns=['Months since last delinquent'])
    train['Credit Score'] = train['Credit Score'].fillna(train['Credit Score'].mean())
    train['Annual Income'] = train['Annual Income'].fillna(train['Annual Income'].mean())
    train['Years in current job'] = train['Years in current job'].fillna(train['Years in current job'].mean())
    train = train.drop(columns=['Loan ID', 'Customer ID'])
    return train

if __name__ == '__main__':
    main()



