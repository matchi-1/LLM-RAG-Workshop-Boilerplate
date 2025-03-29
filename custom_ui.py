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
            background-color: #f7f7f7;    /* whitish */
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
            background-color: #e3f2fd !important;
            padding: 10px;
            border-radius: 10px;
        }

        /* Change input box appearance */
        [data-baseweb="textarea"] {
            font-size: 16px !important;
            padding: 10px !important;
            border: 2px solid #0b7875 !important;
            border-radius: 8px !important;
        }

        /* Style columns */
        [data-testid="column"] {
            box-shadow: rgb(0 0 0 / 20%) 0px 2px 1px -1px, rgb(0 0 0 / 14%) 0px 1px 1px 0px, rgb(0 0 0 / 12%) 0px 1px 3px 0px;
            border-radius: 15px;
            padding: 5% 5% 5% 10%;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)