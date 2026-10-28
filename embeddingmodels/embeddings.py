from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2",
    dimensions=64
)

text = [
    "Hello this is Anshad",
    "I am a developer",
    "I am from kerala"
]

vector =embeddings.embed_documents(text)

print(vector)