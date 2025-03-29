from dotenv import load_dotenv
import os
from typing import List, Optional
import streamlit as st
from langchain.llms.base import LLM
from together import Together

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

