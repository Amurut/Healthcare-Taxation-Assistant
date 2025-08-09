import streamlit as st
from huggingface_hub import InferenceClient
from openai import OpenAI # Import the OpenAI library

llama_model = "meta-llama/Llama-3.2-3B-Instruct"

@st.cache_resource
def get_huggingface_client(api_key):
    """
    Initializes and caches the Hugging Face Inference Client.
    This is isolated to allow for adding other clients (e.g., OpenAI) later.
    """
    return InferenceClient(model=llama_model, token=api_key)

@st.cache_resource
def get_openai_client(api_key):
    """Initializes and caches the OpenAI Client."""
    return OpenAI(api_key=api_key)