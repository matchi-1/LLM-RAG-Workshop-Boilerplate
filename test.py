# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# CHROMA_PATH = "./chroma_db"
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# print(f"Documents stored: {vector_store._collection.count()}")



import os
CHROMA_DB_PATH = "./chroma_db"
import shutil

import os
import shutil
import time
import psutil

CHROMA_PATH = "./chroma_db"

def clear_chroma_db():
    """Deletes all files and folders inside ChromaDB folder"""
    time.sleep(2)  # Give time for processes to release file locks

    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print("✅ ChromaDB folder has been cleared.")
        except Exception as e:
            print(f"⚠️ Error deleting ChromaDB folder: {e}")
    else:
        print("📁 ChromaDB folder does not exist.")

if __name__ == "__main__":
    clear_chroma_db()

