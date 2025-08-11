# modules/llm_clients.py
import streamlit as st
from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
from openai import OpenAI, AuthenticationError

@st.cache_resource
def get_huggingface_client(api_key):
    """Initializes and caches the Hugging Face Inference Client."""
    if not api_key:
        return None
    return InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=api_key)

@st.cache_resource
def get_openai_client(api_key):
    """Initializes and caches the OpenAI Client."""
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def verify_api_key(llm_choice, api_key):
    """
    Verifies if the provided API key is valid for the selected service.
    Returns True if valid, False otherwise, and prints errors to the console.
    """
    if not api_key:
        print("Verification failed: No API key provided.")
        return False
        
    try:
        if "OpenAI" in llm_choice:
            client = OpenAI(api_key=api_key)
            client.models.list()  # A lightweight call to check authentication
            return True
        elif "Llama 3" in llm_choice:
            HfApi().whoami(token=api_key) # Checks if the token is valid
            return True
    except AuthenticationError as e:
        # --- ADDED: Specific logging for OpenAI ---
        print(f"--- OpenAI Authentication Error ---")
        print(f"Failed to authenticate with OpenAI. Please check the API key.")
        print(f"Error Details: {e}")
        print(f"---------------------------------")
        return False
    except HfHubHTTPError as e:
        # --- ADDED: Specific logging for Hugging Face ---
        print(f"--- Hugging Face Authentication Error ---")
        print(f"Failed to authenticate with Hugging Face. Please check the HF Token.")
        print(f"Error Details: {e}")
        print(f"---------------------------------------")
        return False
    except Exception as e:
        # --- ADDED: Catch-all for other unexpected errors (e.g., network) ---
        print(f"--- An Unexpected Error Occurred During Verification ---")
        print(f"Service: {llm_choice}")
        print(f"Error Details: {e}")
        print(f"------------------------------------------------------")
        return False
    
    return False # Should not be reached, but as a fallback