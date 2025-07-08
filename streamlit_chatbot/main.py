import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# --- Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Base System Prompt ---
base_message = {
    "role": "system",
    "content": """
You are a helpful medical assistant. Answer questions related to healthcare, including disease symptoms, diagnosis, treatment, patient rights, and medical policy.

If the question is unrelated to healthcare (such as general knowledge, sports, or entertainment), respond with:
'I’m sorry, I can only answer questions related to healthcare topics.'
"""
}

# --- Initialize Groq Client ---
groq_client = Groq(api_key=groq_api_key)

# --- Convert Chat History ---
def history_to_messages(history):
    messages = [base_message]
    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["bot"]})
    return messages

# --- Chatbot Logic ---
def chatbot_response(user_input, chat_history):
    messages = history_to_messages(chat_history)
    messages.append({"role": "user", "content": user_input})

    try:
        response = groq_client.chat.completions.create(
            # model="llama-3.3-70b-versatile",
            model = "deepseek-r1-distill-llama-70b"
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "❌ Something went wrong. Please try again."

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Secure Healthcare Assistant", page_icon="🩺", layout="wide")

st.sidebar.image(r"M:\techoptima\Guardrails_AI\guardrails_chatbot\data\techoptima-01.jpg", width=180)
st.sidebar.title("🩺 About the Chatbot")
st.sidebar.markdown("""
This AI assistant can help answer healthcare-related questions like:

- Symptoms & Diagnosis  
- Treatments & Medications  
- Medical Rights & Policies

⚠️ Avoid unrelated topics — the assistant will politely decline.
""")

# --- Page Layout ---
col1, col2 = st.columns([2, 1])

# --- Chat History State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Left Panel: Chat UI ---
with col1:
    st.title("🏥 Secure Healthcare Chatbot")

    for entry in st.session_state.chat_history:
        # Plain right-aligned user message
        st.markdown(
            f"""
            <div style='text-align: right; padding: 5px; margin: 5px 0;'>
                {entry["user"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Plain left-aligned bot message
        st.markdown(
            f"""
            <div style='text-align: left; padding: 5px; margin: 5px 0;'>
                {entry["bot"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chat Input
    user_input = st.chat_input("Ask a healthcare-related question...")
    if user_input:
        with st.spinner("Thinking..."):
            response = chatbot_response(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        st.rerun()

# --- Right Panel: Test Prompts ---
# with col2:
#     st.subheader("🧪 Test Questions")

  

#     st.markdown
