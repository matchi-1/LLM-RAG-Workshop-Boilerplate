import streamlit as st

def apply_ui_customization():
    """Applies custom UI styles for the Streamlit app."""
    
    custom_css = """
    <style>
    
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
            font-family: 'Arial', sans-serif !important;     /* HEADERS -- teal */
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

        
        [data-testid="stChatMessage"]:nth-child(odd) [data-testid="stChatMessageContent"] {     
            color: #02717a !important;                               /* BOT CHAT FONT COLORS  -- dark teal */ 
            padding-left: 0.5rem;
        }

        [data-testid="stChatMessage"]:nth-child(even) [data-testid="stChatMessageContent"] {     
            color: white !important;                                 /* USER CHAT FONT COLORS  -- white */ 
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

        [data-testid="stFileUploaderFile"] {
            color: white !important;              /* PDF UPLOADED FILES FONT  --  light grey teal */ 
        }

        [data-testid="stWidgetLabel"] {
            color: #bdd6d9 !important;              /* CAPTION FONT FOR UPLOAD   --  light grey teal*/ 
        }

        [data-testid="stAlertContentWarning"] {
            color: #996600 !important;              /* ALERT FONT  --  white*/ 
        }

        [data-testid="stAlertContentInfo"] {
            color: #b0faff !important;              /* INFO FONT   --  light blue*/ 
        }

        [data-testid="stAlertContentSuccess"] {
            color: #d4fae3 !important;              /* SUCCESS FONT   --  light green*/ 
        }

        [data-testid="stBaseButton-secondary"] {
            color: #0299a6 !important;              /* BUTTON FONT   --  teal*/ 
        }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)