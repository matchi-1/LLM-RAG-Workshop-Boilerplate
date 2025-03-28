from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

print(f"Documents stored: {vector_store._collection.count()}")
