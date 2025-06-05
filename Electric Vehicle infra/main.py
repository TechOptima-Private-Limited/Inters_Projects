import streamlit as st
import requests
import pandas as pd

# Load dataset
df = pd.read_csv("65e18072-d042-4879-ad92-5d6e7ba0cdfd.csv")

# Page config
st.set_page_config(
    page_title="Urban EV Infrastructure Planning Tool",
    layout="wide"
)

# Define 3-column layout
left_col, center_col, right_col = st.columns([0.75, 2, 0.75])

# -------- LEFT COLUMN --------
with left_col:
    st.image("data:image/png;base64,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", use_container_width=False,width=300) 
    st.markdown("<h3 style='color:#FFFFFF;'>Urban EV Infrastructure Planning Tool </h3>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        .css-18e3th9 {
            padding-left: 0 !important;
        }
        .block-container {
            padding-left: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

    st.markdown("### 👥 Target Users")
    st.markdown("""
    - Urban Development Planners 
    - Civil Engineers 
    - EV Charging Companies  
    - Municipal Boards(HMDA) 
    """)

# -------- CENTER COLUMN --------
with center_col:
    st.markdown("#### Select Area and Plan")

    selected_area = st.selectbox("Select Area", sorted(df["area"].dropna().unique()),placeholder="Select an Area")
    num_chargers = st.number_input("Enter number of EV chargers to add", min_value=1, max_value=20, step=1)
    submit = st.button("Submit")

    if submit:
        with st.spinner("OptGPT is Analyzing feasibility..."):
            payload = {
                "area": selected_area,
                "num_chargers": num_chargers
            }
            try:
                response = requests.post("http://localhost:5000/api/ev-plan", json=payload)
                result = response.json()

                if "response" in result:
                    st.markdown("### Analysis:")
                    st.success(result["response"])
                else:
                    st.error("⚠️ Error: " + result.get("error", "Unknown issue."))
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

# -------- RIGHT COLUMN --------
with right_col:
    st.markdown("### ℹ️ About This Tool")
    st.markdown("""
    - 🔍 Assesses whether new EV chargers can be safely added.
    - 🧠 Uses OptGPT analysis to interpret power usage data.
    - 🛠️ Suggests alternatives if selected area is overloaded.
    - ✅ Helps municipalities and planners make data-driven EV infrastructure decisions.
    """)
