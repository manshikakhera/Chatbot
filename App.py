from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define the prompt with corrected syntax
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the queries."),  # Corrected: Comma is present here
        ("user", "Question: {question}")  # Corrected: No comma needed after the last tuple
    ]
)

# Streamlit app title
st.title("Langchain demo with OPENAI API")

# Input field for user question
input_text = st.text_input("Search the topic you want")

# Initialize the language model (ensure the model name is correct and available)
llm = Ollama(model="llama2")

# Output parser
output_parser = StrOutputParser()

# Combine the prompt, model, and parser into a chain
chain = prompt | llm | output_parser

# Display the output when there is input
if input_text:
    st.write(chain.invoke({'question': input_text}))
