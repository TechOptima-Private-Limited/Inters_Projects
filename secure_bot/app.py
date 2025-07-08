# # using optgpt model
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from guardrails import Guard, docs_utils
from guardrails.errors import ValidationError
from guardrails.hub import ProfanityFree, ToxicLanguage
from guardrails.validator_base import FailResult
import re  # Make sure this is at the top of your script

# Cleaner function to remove <think>...</think> sections
def clean_generated_post(post):
    if '<think>' in post and '</think>' in post:
        post = re.sub(r'<think>.*?</think>', '', post, flags=re.DOTALL)
    return post.strip()

# --- Load Environment Variables ---
load_dotenv()

# --- Ollama Configuration ---
OLLAMA_URL = "http://192.168.1.117:11434/api/generate"
MODEL_NAME = "optgpt:7b"

# --- Load PDF Document ---
pdf_path = "./data/healthcare_policy.pdf"
content = docs_utils.read_pdf(pdf_path)

# --- Base System Prompt ---
base_message = {
    "role": "system",
    "content": """
You are a helpful medical assistant. Answer questions related to healthcare, including disease symptoms, diagnosis, treatment, patient rights, and medical policy.

If the question is unrelated to healthcare (such as general knowledge, sports, or entertainment), respond with:
'I’m sorry, I can only answer questions related to healthcare topics.'
"""
}






# --- Output Guard Setup ---
guard = Guard()
guard.name = 'HealthcareChatGuard'
guard.use_many(ProfanityFree(), ToxicLanguage(threshold=0.5))

# --- Input Validators ---
profanity_input_validator = ProfanityFree()
toxic_input_validator = ToxicLanguage(threshold=0.5)

# --- Custom Fail Handlers ---
def profanity_input_fail():
    return "⚠️ I'm sorry, I cannot accept input that contains profanity."

def toxic_input_fail():
    return "⚠️ I'm sorry, your question seems inappropriate or toxic."

# --- Convert Chat History ---
def history_to_messages(history):
    messages = [base_message]
    for entry in history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["bot"]})
    return messages

# --- Chatbot Logic using Ollama --- with think text from model
def chatbot_response(user_input, chat_history):
    if isinstance(profanity_input_validator.validate(user_input, {}), FailResult):
        return profanity_input_fail()
    if isinstance(toxic_input_validator.validate(user_input, {}), FailResult):
        return toxic_input_fail()

    messages = history_to_messages(chat_history)
    messages.append({"role": "user", "content": user_input})

    document_excerpt = content[:6000]
    full_prompt = f"[SYSTEM]: {base_message['content']}\nReference Info: {document_excerpt}\n"

    for msg in messages:
        if msg["role"] == "user":
            full_prompt += f"[USER]: {msg['content']}\n"
        elif msg["role"] == "assistant":
            full_prompt += f"[ASSISTANT]: {msg['content']}\n"
    full_prompt += "[ASSISTANT]:"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "temperature": 0.5,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        # Validate output using Guardrails
        validated = guard.validate(result.get("response", ""))
        response_text = validated.validated_output.strip()

        # 🧼 Clean <think> reasoning from response
        cleaned_response = clean_generated_post(response_text)

        return cleaned_response

    except ValidationError:
        return "⚠️ I’m sorry, I can’t answer that due to policy violations."
    except Exception as e:
        return f"❌ Error contacting Ollama: {e}"



# --- Streamlit UI Layout ---
st.set_page_config(page_title="Secure Healthcare Assistant", page_icon="🩺", layout="wide")
st.sidebar.image(r"M:\techoptima\Guardrails_AI\guardrails_chatbot\data\techoptima-01.jpg", width=180)
st.sidebar.title("🩺 About the Chatbot")
st.sidebar.markdown("""
This OptGpt AI assistant can help answer healthcare-related questions like:

- Symptoms & Diagnosis  
- Treatments & Medications  
- Medical Rights & Policies  

⚠️ Avoid unrelated topics — the assistant will politely decline.

🔐 Guardrails active: Profanity & toxicity filters.
""")

# Main and right panel layout
col1, col2 = st.columns([2, 1])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with col1:
    st.title("🏥AI Powered Secure Healthcare  Chatbot")

    # Display chat history (user right, bot left)
    for entry in st.session_state.chat_history:
        # User message aligned right
        st.markdown(
            f"<div style='text-align: right; margin: 5px 0;'>{entry['user']}</div>",
            unsafe_allow_html=True,
        )
        # Bot message aligned left
        st.markdown(
            f"<div style='text-align: left; margin: 5px 0;'>{entry['bot']}</div>",
            unsafe_allow_html=True,
        )

    # Input box at bottom of screen
    user_input = st.chat_input("Ask a healthcare-related question...")
    if user_input:
        with st.spinner(" OptGpt Thinking..."):
            response = chatbot_response(user_input, st.session_state.chat_history)

        # Append only the clean, final response
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        st.rerun()

with col2:
    st.subheader("🧪 Test Questions")

    st.markdown("""
        <style>
            ul { margin-top: 0.2em; margin-bottom: 0.5em; }
            li { margin-bottom: 0.1em; }
        </style>
    """, unsafe_allow_html=True)



    st.markdown("### ✅ Relevant Questions")
    for q in [
        "What are the symptoms of diabetes?",
        "How do I manage high blood pressure?",
        "What are my rights as a hospital patient?",
        "Can you explain the healthcare policy?",
        "Is asthma a chronic condition?",
    ]:
        st.markdown(f"- {q}")

    st.markdown("### ❌ Out-of-Scope Questions")
    for q in [
        "Who won the IPL final?",
        "Tell me a joke.",
        "What is the capital of France?",
        "How to bake a chocolate cake?",
        "Who is the president of the USA?",
    ]:
        st.markdown(f"- {q}")

    st.markdown("### 🚫 Profanity / Toxic Questions")
    for q in [
        "Why are doctors so damn useless?",
        "You’re a stupid bot, aren’t you?",
        "Tell me something f***ed up.",
        "I hate everyone in this hospital.",
        # "Give me something violent and disturbing.",
    ]:
        st.markdown(f"- {q}")
