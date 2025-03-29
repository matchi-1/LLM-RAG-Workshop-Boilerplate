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
            background-color: white;    /* whitish background */
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
            background-color: #edebeb !important; /* bot color */
        }

        [data-testid="stChatMessage"]:nth-child(even) {
            background: linear-gradient(to right, #54d0d6, #5ca0a3)
        }


        [data-testid="stChatMessageContent"] {
            color: #004347 !important;
            padding-left: 0.5rem;
        }

        [data-testid="stChatMessageAvatarUser"] {
            color: #1fb7bf !important;
            background-color: #e8f9fa !important;
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            color: #5ca0a3 !important;
            background-color: #edebeb !important;
        }

        [data-testid="stHorizontalBlock"] {
            color: #bdd6d9 !important;
        }

        [data-testid="stWidgetLabel"] {
            color: #bdd6d9 !important;
        }


        
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)