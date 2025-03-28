
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from uuid import uuid4


load_dotenv()

# Initialize the models
# models = Models()
# embeddings = models.embeddings_ollama
# llm = models.model_ollama

chunk_size = 1000
chunk_overlap = 50
check_interval = 10


# chroma vector store
vector_store = Chroma (
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",
)


# ingest a file
def ingest_file(file_path):
    # skip non-PDF files
    if not file_path.lower().endswith('.pdf'): 
        print (f"Skipping non-PDF file: {file_path}")
        return

    print(f"Starting to ingest file: {file_path}")
    loader = PyPDFLoader (file_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ""])
    
    documents = text_splitter.split_documents(loaded_documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print(f"Adding {len (documents)} documents to the vector store")
    
    vector_store.add_documents(documents=documents, ids=uuids)
    print("Finished ingesting file: {file_path}")