import streamlit as st
import requests

st.title("Sentiment Analyzer")
st.write('\n')

review = st.text_input("Enter the review", placeholder="Write here...")

if st.button('Predict Sentiment'):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner('OptGPT Sentiment Analyzer is thinking...'):
            try:
                response = requests.post(
                    "http://192.168.1.228:5001/api/sentiment", 
                    json={"review": review},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Sentiment: {data['sentiment'].capitalize()}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
