import streamlit as st
import google.generativeai as genai

# 🔐 Configure Gemini API with your API key
genai.configure(api_key="AIzaSyAXZxpnNAxyKJYV6Xx-4bzhUPo4YCcrjT4")

# Use Gemini 1.5 Flash model
FLASH_MODEL = "models/gemini-1.5-flash-latest"

# Custom system prompt tailored to Indian early-stage investors
system_prompt = (
    "You are a helpful assistant for early-stage startup investors in India. "
    "Respond in clear, simple terms and explain key concepts related to angel investing, SEBI regulations, "
    "SAFE notes, valuation techniques, term sheets, Indian taxation policies like Angel Tax, and startup exits. "
    "Use real or realistic examples from the Indian startup ecosystem when relevant. Be conversational and educational."
)

# Set up Streamlit page
st.set_page_config(page_title="Investor Chatbot 🇮🇳", page_icon="📈")
st.title("🇮🇳 Indian Startup Investor Chatbot")
st.caption("Powered by Gemini 1.5 Flash — Ask about startup funding, SEBI rules, Angel Tax, or anything else!")

# Initialize chat session
if "chat" not in st.session_state:
    model = genai.GenerativeModel(FLASH_MODEL)
    st.session_state.chat = model.start_chat(history=[
        {"role": "user", "parts": [system_prompt]}
    ])
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Namaste! I'm here to help you navigate early-stage investing in Indian startups. What would you like to know?"
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask about SEBI rules, SAFE notes, valuations, exits, or Angel Tax...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from Gemini Flash
    try:
        response = st.session_state.chat.send_message(user_input)
        reply = response.text
    except Exception as e:
        reply = f"⚠️ Gemini Flash Error: {e}"

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
