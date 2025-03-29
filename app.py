import os
import psutil
import keyboard
import time
import streamlit as st
from dotenv import load_dotenv
#from PyPDF2 import PdfReader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceInstructEmbeddings     #, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_huggingface import HuggingFaceEmbeddings    # uncomment to use huggingface
#from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from together import Together
from langchain.llms.base import LLM
from typing import List, Optional
from langchain_chroma import Chroma

from ingest import ingest_file, DATA_FOLDER, CHROMA_PATH, TogetherEmbeddings   #   vector_store,    import the the ingest_file method and vector store from ingest.py
from pdf_utils import list_pdfs, delete_pdf


load_dotenv()  # load environment variables from .env file
together_api_key = os.getenv("TOGETHER_LLM_API_KEY") # Together api key: get the API key from the environment variable


class TogetherLLM(LLM):
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    api_key: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # define system instructions -- acts as the bot's main instructions and personality
        system_instructions = """
        You are James, the fun AI assistant that answers the user based on retrieved document context.
        You should always be enthusiastic and helpful when replying.
        
        If the question is general knowledge and isn't necessarily related to the context, try and answer it still based on facts.
       
        
        """ # you can make this longer and more specific to your use case

        # If you are unsure of the answer, just say 'I don't know based on the given documents.  --- you can also add this

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
            max_tokens=1000,      # limit response length
            temperature=0.5,       # creativity (Lower = More Deterministic)
            top_p=0.5,             # nucleus Sampling (Lower = Focused, Higher = Diverse)
            repetition_penalty=1.1 # penalize repetitive words
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

# helper function to stream text output
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

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

    # display bot's response in markdown
    with st.chat_message("assistant"):
        st.write_stream(stream_data(bot_reply))

    # add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Print chat history to terminal
    print("Chat History:")
    print(str(st.session_state.chat_history))


def main():
    load_dotenv()
    st.set_page_config(page_title="ChatTGP", page_icon="💊")  # chatbot name

    # initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "chroma_reset_needed" not in st.session_state:
        st.session_state.chroma_reset_needed = False

    # initialize chat history if not already set
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! How can I assist you today?"}
        ]

    st.header("ChatTGP 👩🏼‍⚕️")

    # display chat history (persists across reruns)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input field
    if user_question := st.chat_input("Ask a question about your documents..."):
        handle_userinput(user_question)

    with st.sidebar:
        st.header("📄 Retrieval Augmented Generation (RAG)")
        
        # upload new PDFs
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        
        # logic to process uploaded PDFs
        if st.button("Process"):
            if not pdf_docs:
                st.warning("⚠️ No files uploaded. Please select at least one PDF.")
                time.sleep(2)  # allow warning to persist
            else:
                existing_files = {pdf["filename"].lstrip("_") for pdf in list_pdfs()}  # get base filenames
                valid_files = []  # store valid PDFs for processing

                for pdf in pdf_docs:
                    base_name = pdf.name.lstrip("_")
                    if base_name in existing_files:
                        st.error(f"❌ File '{pdf.name}' already exists. Please rename or upload a different file.")
                        time.sleep(2)
                    elif not pdf.name.endswith(".pdf"):
                        st.error(f"❌ Invalid file format: '{pdf.name}'. Only PDFs are allowed.")
                        time.sleep(2)
                    else:
                        valid_files.append(pdf)

                if valid_files:  # process only valid PDFs
                    for pdf in valid_files:
                        file_path = os.path.join(DATA_FOLDER, pdf.name)
                        with st.spinner(f"Processing '{pdf.name}'..."):
                            try:
                                with open(file_path, "wb") as f:
                                    f.write(pdf.read())

                                # process the PDF and add to Chroma
                                ingest_file(file_path)
                                st.success(f"✅ Successfully processed: {pdf.name}")
                            except Exception as e:
                                st.error(f"⚠️ Error processing '{pdf.name}': {e}")
                                time.sleep(2)

                    time.sleep(2)  # allow success messages to persist
                    st.rerun()  # refresh UI

                else:
                    st.info("No valid PDFs to process.")
       
        st.divider()
        # display stored PDFs in a table
        st.subheader("📋 Processed PDFs in ChromaDB")
        pdf_list = list_pdfs()
        
        if pdf_list:
            for pdf in pdf_list:
                col1, col2 = st.columns([3, 1])  # table columns

                col1.write(f"{pdf['name']}")  # processed status and PDF name

                # Store the confirmation state in session_state
                delete_confirm_key = f"delete_confirm_{pdf['filename']}"

                if delete_confirm_key not in st.session_state:
                    st.session_state[delete_confirm_key] = False  # Default to no confirmation

                # If confirmation is needed, show a warning and buttons
                if col2.button("🗑️", key=pdf["filename"]):
                    # Toggle the confirmation state
                    st.session_state[delete_confirm_key] = not st.session_state[delete_confirm_key]

                # Handle deletion confirmation and cancellation
                if st.session_state[delete_confirm_key]:
                    st.warning(f"Are you sure you want to delete **{pdf['filename']}**?")

                    col_confirm, col_cancel = st.columns([1, 1])

                    with col_confirm:
                        if st.button("✅ Yes, Delete", key=f"confirm_{pdf['filename']}"):
                            delete_pdf(pdf["filename"])
                            st.session_state["chroma_reset_needed"] = True
                            st.session_state[delete_confirm_key] = False  # Reset confirmation state
                            st.rerun()  # refresh the UI after deletion

                    with col_cancel:
                        if st.button("❌ Cancel", key=f"cancel_{pdf['filename']}"):
                            st.session_state[delete_confirm_key] = False  # Reset confirmation state
                            st.rerun() 

        
        st.divider()
        st.markdown(
            '<p style="font-size: 12px; color: grey;">'
            'Note: If you deleted a file here and have not reset ChromaDB yet, the embeddings still exist but are not visible in the list. Terminate the streamlit app and delete the "./chroma_db" folder to reset, then you can  run the app again.'
            '</p>',
            unsafe_allow_html=True
        )


    # initialize embeddings
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = TogetherEmbeddings()

    # initialize a Chromadb vector store
    vectorstore = Chroma(collection_name = "documents",
                        persist_directory=CHROMA_PATH,
                        embedding_function=embeddings)

    # store conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)


    # show warning and reset button if a file has been deleted
    if st.session_state.get("chroma_reset_needed", False):
        st.warning(
            "⚠️ Some deleted PDFs still have embeddings stored in ChromaDB. "
            "To fully reset the knowledge base, follow these steps:\n\n"
            "1️⃣ Click the **Exit Streamlit App** button below to terminate the current session.\n\n"
            "2️⃣ Open your IDE and **delete the `chroma_db` directory** manually:\n\n"
            "3️⃣ Restart the Streamlit app by running:\n"
            "   ```sh\n"
            "   streamlit run app.py\n"
            "   ```\n\n"
            "🔄 This ensures ChromaDB is fully cleared and updated."
        )

        if st.button("❌ Exit Streamlit App"):
            st.write("🔄 Exiting... Please delete 'chroma_db' folder and restart the app.")
            time.sleep(4)
            # close streamlit browser tab
            keyboard.press_and_release('ctrl+w')
            # terminate streamlit python process
            pid = os.getpid()
            p = psutil.Process(pid)
            p.terminate()

if __name__ == '__main__':
    main()