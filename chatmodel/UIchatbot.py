from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

st.set_page_config(page_title="AI Personality Chat", page_icon="🤖", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 2.5rem; }
    .hero { text-align: center; margin-bottom: 1.5rem; }
    .hero h1 { font-size: 2.4rem; margin-bottom: 0.2rem; }
    .hero p { color: #8a8f98; font-size: 1.05rem; }
    .persona-badge {
        display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
        background: rgba(124, 92, 255, 0.15); color: #7c5cff;
        font-size: 0.85rem; font-weight: 600;
    }
    div.stButton > button {
        border-radius: 12px; font-weight: 600; padding: 0.6rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Personalities ----------
PERSONALITIES = {
    "😂 Funny": "you are funny AI agent",
    "😏 Sarcastic": "you are a sarcastic AI agent with dry wit",
    "🧙 Wise Mentor": "you are a calm, wise mentor who gives thoughtful advice",
    "🏴‍☠️ Pirate": "you are a pirate AI agent who talks like a pirate",
    "🎓 Friendly Teacher": "you are a friendly teacher who explains things simply",
    "✍️ Custom": None,
}


@st.cache_resource
def get_model():
    return init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq"
    )


model = get_model()

# ---------- Step 1: choose personality ----------
if "persona" not in st.session_state:
    st.markdown(
        '<div class="hero"><h1>🤖 Choose AI Personality</h1>'
        "<p>Pick how your AI should talk, then start chatting.</p></div>",
        unsafe_allow_html=True,
    )

    choice = st.radio(
        "Personality",
        list(PERSONALITIES.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    custom_text = ""
    if PERSONALITIES[choice] is None:
        custom_text = st.text_area(
            "Describe your personality",
            placeholder="e.g. you are a grumpy cat who loves coding",
        )

    if st.button("Start chat 🚀", use_container_width=True, type="primary"):
        persona = custom_text.strip() if PERSONALITIES[choice] is None else PERSONALITIES[choice]
        if not persona:
            st.warning("Please describe the personality first.")
        else:
            st.session_state.persona = persona
            st.session_state.persona_label = choice
            st.session_state.messages = [SystemMessage(content=persona)]
            st.rerun()
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Current personality")
    st.markdown(
        f'<span class="persona-badge">{st.session_state.persona_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(st.session_state.persona)
    st.divider()
    if st.button("🔄 Change personality", use_container_width=True):
        for key in ("persona", "persona_label", "messages"):
            st.session_state.pop(key, None)
        st.rerun()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=st.session_state.persona)]
        st.rerun()
    st.caption("Type 0 in the chat to exit the app.")

# ---------- Step 2: chat ----------
st.markdown(
    '<div class="hero"><h1>💬 Chat</h1></div>',
    unsafe_allow_html=True,
)

for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.write(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(m.content)

prompt = st.chat_input("Type your message...")

if prompt:
    if prompt == "0":
        st.info("Application closed.")
        os._exit(0)

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
        res = response.content
        st.write(res)

    st.session_state.messages.append(AIMessage(content=res))