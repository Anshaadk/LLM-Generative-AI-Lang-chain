from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from langchain.chat_models import init_chat_model

# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq"
)
print("chose AI persanality")
persenality = input("write here")

messages = [
    SystemMessage(content=persenality)
]

print("_____________welcome type 0 exit application__________")
while True:
    prompt  = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response = model.invoke(messages)
    res = response.content
    messages.append(AIMessage(content=res))

    print("bot :",res)