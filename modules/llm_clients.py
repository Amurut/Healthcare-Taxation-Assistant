# modules/llm_clients.py
import streamlit as st
from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
from openai import OpenAI, AuthenticationError
import json

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
    Verifies API key. Returns a tuple: (is_valid: bool, message: str).
    Also prints the full error to the browser console for debugging.
    """
    if not api_key:
        return False, "Error: No API key provided."
        
    try:
        if "OpenAI" in llm_choice:
            client = OpenAI(api_key=api_key)
            client.models.list()
            return True, "✅ OpenAI key is valid!"
        elif "Llama 3" in llm_choice:
            HfApi().whoami(token=api_key)
            return True, "✅ Hugging Face token is valid!"
            
    except Exception as e:
        # --- NEW: Logic to print the full error to the browser console ---
        error_message_for_ui = "❌ Authentication Failed. Check browser console (F12) for details."
        
        # Format the full error for JavaScript
        full_error_str = repr(e).replace('`', "'").replace('\\', '\\\\')
        js_safe_error = json.dumps(full_error_str)

        # Inject JavaScript to log the error
        st.html(f"<script>console.error('API Key Verification Error for {llm_choice}:', {js_safe_error});</script>")
        
        # For the UI, we can return a simpler message or a specific part of the error
        if isinstance(e, AuthenticationError):
            error_message_for_ui = f"❌ OpenAI Auth Error: {e.body.get('message', 'Invalid key.')}"
        elif isinstance(e, HfHubHTTPError):
            error_message_for_ui = f"❌ HF Auth Error: {str(e)}"
        
        return False, error_message_for_ui
    
    return False, "❌ Verification failed due to an unknown error."