import os
#from langchain_huggingface import HuggingFaceEmbeddings # uncomment to use huggingface
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from together import Together
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

DATA_FOLDER = "./data"
CHROMA_PATH = "./chroma_db"

load_dotenv()  # load environment variables from .env file
together_api_key = os.getenv("TOGETHER_LLM_API_KEY") # Together api key: get the API key from the environment variable

# # set up embeddings 
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")    

class TogetherEmbeddings(Embeddings):
    """Custom LangChain embedding class using Together AI"""
    
    def __init__(self, model_name="togethercomputer/m2-bert-80M-32k-retrieval"):
        self.client = Together(api_key=together_api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        """Embed a list of documents"""
        responses = [self.client.embeddings.create(model=self.model_name, input=text) for text in texts]
        return [res.data[0].embedding for res in responses]

    def embed_query(self, text):
        """Embed a single query"""
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding

# Replace HuggingFaceEmbeddings with Together AI embeddings
embeddings = TogetherEmbeddings()


# initialize a Chromadb vector store
vector_store = Chroma(collection_name = "documents",
                    persist_directory=CHROMA_PATH,
                    embedding_function=embeddings)

def ingest_file(pdf_path):
    """Ingest a PDF file into ChromaDB"""
    
    # load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if not pages:
        print(f">>>>>>> ERROR: No text found in {pdf_path}")
        return

    # print extracted content for debugging
    print(f"========= Extracted {len(pages)} pages from {pdf_path}")
    #print(f"========= First Page Sample:\n{pages[0].page_content[:500]}")  # Print first 500 chars

    # split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    if not docs:
        print(f">>>>>>> ERROR: No text chunks created for {pdf_path}")
        return

    # print chunk info
    print(f"========= {len(docs)} chunks created for {pdf_path}")

    # add documents
    vector_store.add_documents(docs)

    # rename file to mark as processed
    os.rename(pdf_path, os.path.join(DATA_FOLDER, "_" + os.path.basename(pdf_path)))

    print(f"\n\n========= {pdf_path} successfully ingested and stored!")



if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf") and not file.startswith("_"):
            ingest_file(os.path.join(DATA_FOLDER, file))
