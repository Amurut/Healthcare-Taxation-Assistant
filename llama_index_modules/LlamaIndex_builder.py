# llama_index_modules/LlamaIndex_builder.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json

# --- Core Configuration ---
# Use the same embedding model as your custom code for a fair comparison
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
# The LLM can be configured here, but we'll primarily define it in the agent
Settings.llm = OpenAI(model="gpt-4o")

# --- Index Building Functions ---
def build_irs_index(urls_filepath="publications.json", save_dir="llama_index_stores/irs_index"):
    """Builds and saves a LlamaIndex VectorStoreIndex from IRS web pages."""
    if not os.path.exists(urls_filepath):
        print(f"URL file not found: {urls_filepath}")
        return
        
    with open(urls_filepath, 'r') as f:
        urls_data = json.load(f)
    urls = list(urls_data.values())
    
    # Load data from web
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=urls)
    
    # Create and persist the index
    print(f"Creating LlamaIndex for {len(documents)} IRS documents...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=save_dir)
    print(f"IRS index saved to '{save_dir}'")

def build_cases_index(pdf_dir="source_documents/legal_cases", save_dir="llama_index_stores/cases_index"):
    """Builds and saves a LlamaIndex VectorStoreIndex from local PDFs."""
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return

    # Load data from PDFs
    loader = SimpleDirectoryReader(pdf_dir)
    documents = loader.load_data()

    if not documents:
        print(f"No documents found in {pdf_dir}.")
        return

    # Create and persist the index
    print(f"Creating LlamaIndex for {len(documents)} legal case documents...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=save_dir)
    print(f"Legal cases index saved to '{save_dir}'")