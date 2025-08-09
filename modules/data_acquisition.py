# data_acquisition.py
import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import os

def load_urls_from_file(filepath="publications.json"):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: The configuration file {filepath} was not found.")
        return {}

def scrape_publication(name, url):
    try:
        headers = {'User-Agent': 'Modular-RAG-Legal-Interpreter/1.0'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content_area = (
            soup.find('main', id='main-content') or
            soup.find('div', class_='usa-prose') or
            soup.find('article') or
            soup.find('div', id='content') or
            soup.find("div", {"id": "bodytext"}) or
            soup
            )
        if content_area:
                # Remove non-relevant elements like forms, scripts, etc.
            for element in content_area(['script', 'style', 'form', 'nav']):
                element.decompose()
            return content_area.get_text(separator='\n', strip=True)
        st.sidebar.warning(f"Could not find main content for {name}.", icon="⚠️")
        return None
    except requests.exceptions.RequestException as e:
        st.sidebar.warning(f"Could not scrape {name}: {e}", icon="⚠️")
        return None

def extract_text_from_pdfs(pdf_folder_path):
    """
    Extracts text from all PDF files in a given folder.
    Returns a dictionary where keys are filenames and values are the extracted text.
    """
    pdf_texts = {}
    if not os.path.exists(pdf_folder_path):
        st.sidebar.warning(f"PDF folder not found at: {pdf_folder_path}")
        return pdf_texts
        
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            try:
                path = os.path.join(pdf_folder_path, filename)
                with fitz.open(path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    pdf_texts[filename] = text
            except Exception as e:
                st.sidebar.error(f"Failed to read {filename}: {e}")
    return pdf_texts