import streamlit as st
from huggingface_hub import InferenceClient, HfApi
# Correctly import exceptions from the 'huggingface_hub.utils' submodule
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
from openai import OpenAI, AuthenticationError

@st.cache_resource
def get_huggingface_client(api_key):
    """
    Initializes and caches the Hugging Face Inference Client for Llama 3.
    """
    if not api_key:
        return None
    return InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=api_key)

@st.cache_resource
def get_openai_client(api_key):
    """
    Initializes and caches the OpenAI Client.
    """
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def verify_api_key(llm_choice, api_key):
    """
    Verifies if the provided API key is valid for the selected service.
    Returns True if valid, False otherwise.
    """
    if not api_key:
        return False
        
    try:
        if "OpenAI" in llm_choice:
            client = OpenAI(api_key=api_key)
            client.models.list()  # A lightweight call to check authentication
            return True
        elif "Llama 3" in llm_choice:
            HfApi().whoami(token=api_key) # Checks if the token is valid
            return True
    # The 'except' block now catches the exceptions from their correct import path
    except (AuthenticationError, ValueError, GatedRepoError, RepositoryNotFoundError, HfHubHTTPError):
        return False
    except Exception:
        # Catch other potential network errors etc.
        return False
    return False