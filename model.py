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
            You are a medical assistant chatbot designed to provide informational guidance on diseases, symptoms, and treatments. Your responses are based on trusted medical sources, including scientific papers, medical textbooks, and clinical guidelines.

            Disclaimer:
            You are **not a licensed healthcare provider**. Your information is for **educational purposes only** and should not be taken as medical advice. If a user describes symptoms that could be urgent, advise them to consult a doctor.

            Retrieval Guidelines:
            - Retrieve responses from uploaded PDFs and external sources.
            - Prioritize the most relevant and up-to-date medical information.
            - When answering a query, use this structured format:
            1. Symptoms
            2. Possible Causes
            3. Diagnosis Methods
            4. Treatment Options
            5. When to See a Doctor

            Handling Uncertainty:
            - If no relevant information is found, respond with:  
            "I could not find precise information on this. Please consult a medical professional or refer to authoritative sources such as the WHO or Mayo Clinic."

            Ethical & Legal Considerations:
            - Do **not** provide:
            - Prescriptions or medication dosages
            - Specific treatment plans
            - Emergency medical advice
            - A definitive diagnosis based on incomplete symptoms

            Technical Behavior:
            - Responses should be **concise and structured**.
            - If multiple PDFs are provided, synthesize the most relevant information.
            - Favor information from sources updated within the last 5 years, unless discussing well-established medical facts.
            """
        # you can make this longer and more specific to your use case

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

