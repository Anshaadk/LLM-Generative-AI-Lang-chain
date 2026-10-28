from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name ="sentence-transformers/all-MiniLM-L6-v2",
)


text = [
    "Hello this is Anshad",
    "I am a developer",
    "I am from kerala"
]

vector =embeddings.embed_documents(text)

print(vector)