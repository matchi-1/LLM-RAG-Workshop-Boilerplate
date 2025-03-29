import streamlit as st

def apply_ui_customization():
    """Applies custom UI styles for the Streamlit app."""
    
    custom_css = """
    <style>
        /* Customize sidebar */
        [data-testid="stSidebar"] {
            background-color: #00858a !important;   /* dark teal */
            padding: 1rem;
        }

        /* Style main content area */
        [data-testid="stAppViewContainer"] {
            background-color: #f7f7f7;    /* whitish background */
        }

        /* Style headers */
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            font-family: 'Arial', sans-serif !important;
            color: white !important;
        }

        /* Customize chat messages */
        [data-testid="stChatMessage"] {
            background-color: #9edbde !important;
            padding: 10px;
            border-radius: 10px;
            color: black !important;
        }

        
        /* Alternate background colors */
        [data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #9edbde !important; /* Light blue for user */
        }

        [data-testid="stChatMessage"]:nth-child(even) {
            background-color: #f8d7da !important; /* Light red for bot */
        }

        
        [data-testid="stChatMessageContent"] {
            color: black !important;
        }

        [data-testid="stChatMessageAvatarUser"] {
            color: black !important;
            background-color: #00b7db !important;00aeb8
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            color: black !important;
            background-color: #f2b600 !important;
        }

        
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)