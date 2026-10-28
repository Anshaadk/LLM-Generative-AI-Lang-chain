from dotenv import load_dotenv

load_dotenv()

import os
from langchain.chat_models import init_chat_model
# from langchain_groq import ChatGroq

# google model

model = init_chat_model(
    model="gemini-3-flash-preview",
    model_provider="google_genai",
    temperature=0.9,
    max_tokens=100
    )
# model = ChatGroq(model="openai/gpt-oss-120b")

response = model.invoke("what is the football laliga?")
print(response)