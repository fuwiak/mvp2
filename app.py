# backend/app.py
import streamlit as st
import pdfplumber
import io
import json
import os
import tempfile
from typing import List
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Support Chatbot MVP")

st.sidebar.title("Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

knowledge_base = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                knowledge_base += page.extract_text() + "\n"

# Persistent state for additional prompt
if "additional_prompt" not in st.session_state:
    st.session_state["additional_prompt"] = ""

def set_prompt():
    st.session_state["additional_prompt"] = st.session_state["prompt_input"]

st.sidebar.text_area("Additional Prompt", key="prompt_input")
st.sidebar.button("Apply Prompt", on_click=set_prompt)

def ask_llm(question: str, context: str, additional_prompt: str):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system", 
                "content": """You are an assistant that answers questions based on the uploaded PDF document. 
                Use only the information provided in the given context. 
                If there is not enough information, state so."""
            },
            {
                "role": "user", 
                "content": f"""Context of the document:
                {context}

                Additional instructions:
                {additional_prompt}

                Question: {question}

                Please respond based only on the provided context."""
            }
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    return response.choices[0].message.content

st.title("Support Chatbot")
user_input = st.text_input("Ask a question")

if st.button("Submit"):
    if user_input:
        response = ask_llm(user_input, knowledge_base, st.session_state["additional_prompt"])
        st.write(response)
    else:
        st.warning("Please enter a question.")