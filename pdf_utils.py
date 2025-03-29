import os
import time
import streamlit as st

DATA_FOLDER = "./data"

def list_pdfs():
    """Lists all PDFs in the data folder with their processed status."""
    pdfs = []
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)  # ensure data directory exists
    
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            is_processed = filename.startswith("_")  # processed if prefixed with "_"
            display_name = filename.lstrip("_")  # remove "_" prefix for UI display
            pdfs.append({"name": display_name, "processed": is_processed, "filename": filename})
    return pdfs


def delete_pdf(filename):
    """Deletes a PDF file, resets ChromaDB, and reprocesses remaining PDFs."""
    file_path = os.path.join(DATA_FOLDER, filename)

    if os.path.exists(file_path):
        os.remove(file_path)  # remove file
        st.success(f"✅ Deleted: {filename}")
        st.warning(
            "⚠️ A file was deleted, but its data is still in ChromaDB."
        )
        time.sleep(3) 
