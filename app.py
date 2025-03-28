import os
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
from htmlTemplates import css, bot_template, user_template
#from langchain_community.llms import HuggingFaceHub
from together import Together
from langchain.llms.base import LLM
from typing import List, Optional
from langchain_chroma import Chroma

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

from ingest import ingest_file, vector_store,DATA_FOLDER, CHROMA_PATH   # import the the ingest_file method and vector store from ingest.py

load_dotenv()  # load environment variables from .env file
together_api_key = os.getenv("TOGETHER_LLM_API_KEY") # Together api key: get the API key from the environment variable





# # define the retrieval chain
# retriever = vector_store.as_retriever(kwargs={"k": 10})
# combine_docs_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=3000,
#         chunk_overlap=200,
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     #embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore



# Prompt Template with system instructions
template = """You are James the fun AI assistant that answers the user based on retrieved document context.
            If the information is not available, do not make up an answer.

            Context:
            {context}

            Chat History:
            {chat_history}

            User Query:
            {query}

            Answer:
            """


prompt = PromptTemplate.from_template(template)

class TogetherLLM(LLM):
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    api_key: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Define system instructions with context
        system_instructions = """
        You are John, the fun AI assistant that answers the user based on retrieved document context.
        If the information is not available, do not make up an answer. Instead, say 'I don't know based on the given documents.'
        """

        template = """
        Context:
        {context}

        Chat History:
        {chat_history}

        User Query:
        {query}

        Answer:
        """

        # Retrieve chat history from memory
        chat_history = "\n".join(
            [f"{m.content}" for m in st.session_state.chat_history]
        ) if st.session_state.chat_history else "No previous conversation."

        # Replace placeholders in the template
        full_prompt = template.format(
            context="No relevant context found yet.",  # This will be replaced later
            chat_history=chat_history,
            query=prompt
        )

        # Call Together AI's model
        client = Together(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_instructions},
                      {"role": "user", "content": full_prompt}],
            max_tokens=1000  # Adjust token limit as needed
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
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User:** {message.content}")
        else:
            st.write(f"**Bot:** {message.content}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    print(f">>>>>>>>>>>>>>>>>>>>>> user_query: {user_question}")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                for pdf in pdf_docs:
                    file_path = os.path.join("./data", pdf.name)
                    with open(file_path, "wb") as f:
                        f.write(pdf.read())
                    
                    # process the PDF and add to Chroma
                    ingest_file(file_path)

    # load the stored vector database
    vectorstore = vector_store  # directly use Chroma from ingest.py

    # store conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()