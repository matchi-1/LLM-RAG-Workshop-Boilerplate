import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceInstructEmbeddings     #, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from together import Together
from langchain.llms.base import LLM
from typing import List, Optional
from langchain_chroma import Chroma

from ingest import ingest_file, vector_store   # import the the ingest_file method and vector store from ingest.py

load_dotenv()  # load environment variables from .env file
together_api_key = os.getenv("TOGETHER_LLM_API_KEY") # Together api key: get the API key from the environment variable


class TogetherLLM(LLM):
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    api_key: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # define system instructions -- acts as the bot's main instructions and personality
        system_instructions = """
        You are James, the fun AI assistant that answers the user based on retrieved document context.
        If the information is not available, do not make up an answer. Instead, say 'I don't know based on the given documents.'
        """ # you can make this longer and more specific to your use case

        # retrieve chat history from memory
        chat_history = "\n".join(
            [f"{m.content}" for m in st.session_state.chat_history]
        ) if st.session_state.chat_history else "No previous conversation."


        template = """
        Answer it based on this context:
        {context}

        This is the chat History:
        {chat_history}

        This is the current user query:
        {query}

        Answer:
        """     # you can also modify this template to change the bot's response style for answering individual queries

        # replace placeholders in the template
        full_prompt = template.format(
            context="No relevant context found yet.",  # this will be replaced later
            chat_history=chat_history,
            query=prompt
        )

        # call Together AI's model
        client = Together(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_instructions},
                      {"role": "user", "content": full_prompt}],
            max_tokens=1000  # adjust token limit as needed
        )

        return response.choices[0].message.content


    @property
    def _llm_type(self) -> str:
        return "together_ai"


def get_conversation_chain(vectorstore):
    together_llm = TogetherLLM(api_key=together_api_key)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=together_llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        # combine_docs_chain=combine_docs_chain
    )

    return conversation_chain


def handle_userinput(user_question):
    """Handles user input, gets a response from the chatbot, and updates chat history."""

    # display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)

    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})

    # get chatbot response from the conversation chain
    response = st.session_state.conversation.invoke({"question": user_question})
    bot_reply = response["answer"] 
    st.session_state.chat_history = response["chat_history"]

    # stream bot's response dynamically
    with st.chat_message("assistant"):
        bot_container = st.empty()
        full_response = ""
        for word in bot_reply.split():
            full_response += word + " "
            bot_container.markdown(full_response)
            time.sleep(0.05)  # Simulate streaming delay

    # add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


DATA_DIR = "./data"

def list_pdfs():
    """Lists all PDFs in the data folder with their processed status."""
    pdfs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)  # Ensure data directory exists
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            is_processed = filename.startswith("_")  # Processed if prefixed with "_"
            display_name = filename.lstrip("_")  # Remove "_" prefix for UI display
            pdfs.append({"name": display_name, "processed": is_processed, "filename": filename})
    return pdfs

def delete_pdf(filename):
    """Deletes a PDF from the data folder."""
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success(f"Deleted {filename}")

def main():
    load_dotenv()
    st.set_page_config(page_title="BOTTIE", page_icon="👒")  # Chatbot name

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Store chat messages for UI

    st.header("Ask BOTTIE a question 👒")

    # Display chat history (persists across reruns)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field
    if user_question := st.chat_input("Ask a question about your documents..."):
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📄 Your Documents")
        
        # Upload new PDFs
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                for pdf in pdf_docs:
                    file_path = os.path.join(DATA_DIR, pdf.name)
                    with open(file_path, "wb") as f:
                        f.write(pdf.read())
                    
                    # Process the PDF and add to Chroma
                    ingest_file(file_path)

        # Display stored PDFs in a table
        st.subheader("📋 Uploaded PDFs")
        pdf_list = list_pdfs()
        
        if pdf_list:
            for pdf in pdf_list:
                col1, col2 = st.columns([3, 1])  # Table columns

                col1.write(f"{pdf['name']}")  # processed status and PDF name
                
                # Delete button
                if col2.button("🗑️", key=pdf["filename"]):
                    delete_pdf(pdf["filename"])
                    st.rerun()  # Refresh the UI after deletion

    # Load vector store
    vectorstore = vector_store  # Directly use Chroma from ingest.py

    # Store conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()