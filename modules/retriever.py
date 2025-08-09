# modules/retriever.py
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache_resource
def get_embedding_model():
    # ... (this function remains the same)
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(source_content, chunk_size=1500, chunk_overlap=200):
    """
    A generic function to take a dictionary of texts, chunk them,
    and return the chunks and a ready-to-use FAISS index.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for source, text in source_content.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({'source': source, 'text': chunk})
            
    if not all_chunks:
        return None, None

    embedding_model = get_embedding_model()
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    
    chunk_embeddings = embedding_model.encode(chunk_texts, convert_to_tensor=False)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')

    embedding_dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(chunk_embeddings)
    
    return all_chunks, index

def retrieve_context(query, chunks, index, top_k=3):
    """
    A generic function to retrieve context from a given FAISS index.
    """
    if index is None:
        return "No knowledge base available for this tool.", []

    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, top_k)
    
    context = ""
    sources = set()
    for idx in indices[0]:
        if idx != -1:
            chunk_info = chunks[idx]
            sources.add(chunk_info['source'])
            context += f"--- Context from: {chunk_info['source']} ---\n"
            context += f"{chunk_info['text']}\n\n"
            
    return context, list(sources)