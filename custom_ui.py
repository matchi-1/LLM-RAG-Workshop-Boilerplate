import streamlit as st

def apply_ui_customization():
    """Applies custom UI styles for the Streamlit app."""
    
    custom_css = """
    <style>
        /* Customize sidebar */
        [data-testid="stSidebar"] {
            background-color: #038691 !important;   /* SIDEBAR -- dark blue teal */
            padding: 1rem;
        }

        /* Style main content area */
        [data-testid="stAppViewContainer"] {
            background-color: white;                    /* APP BG WHITE */
        }

        [data-testid="stMarkdownContainer"] h3 {
            font-family: 'Arial', sans-serif !important;     /* SUBHEADERS -- white */
            color: white !important;
        }

        [data-testid="stMarkdownContainer"] h2{
            font-family: 'Arial', sans-serif !important;     /* HEADERS -- white */
            color: #0299a6 !important;
        }

        /* Customize chat messages */
        [data-testid="stChatMessage"] { 
            background-color: #9edbde !important;         /* CHAT DIVS -- common styles */
            padding: 10px;
            border-radius: 10px; 
        }

        
        /* Alternate background colors */
        [data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #edebeb !important;           /* BOT DIV  -- light grey */  
        }

        [data-testid="stChatMessage"]:nth-child(even) {
            background: linear-gradient(to right, #54d0d6, #0094a1)    /* USER DIV  -- gradient teal blue */  
        }


        [data-testid="stChatMessageContent"] {     /* CHAT FONT COLORS  -- dark teal */ 
            color: #047d87 !important;
            padding-left: 0.5rem;
        }

        [data-testid="stChatMessageAvatarUser"] {
            color: #1fb7bf !important;
            background-color: #e8f9fa !important;      /* AVATAR ICON  -- white - teal */ 
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            color: #5ca0a3 !important;
            background-color: transparent;    /* ASSISTANT ICON  --  transparent - teal */ 
        }

        [data-testid="stHorizontalBlock"] {
            color: #bdd6d9 !important;              /* PDF UPLOADS FONT  --  light grey teal */ 
        }

        [data-testid="stWidgetLabel"] {
            color: #bdd6d9 !important;              /* CAPTION FONT FOR UPLOAD   --  light grey teal*/ 
        }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)