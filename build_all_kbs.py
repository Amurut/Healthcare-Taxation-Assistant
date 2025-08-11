# build_all_kbs.py
import os
from modules import data_acquisition as custom_da, retriever as custom_retriever
from llama_index_modules import LlamaIndex_builder
import faiss
import pickle

print("🚀 Starting Unified Knowledge Base build process...")
os.makedirs("knowledge_stores", exist_ok=True)
os.makedirs("llama_index_stores", exist_ok=True)
os.makedirs("debug_outputs", exist_ok=True)
print("✅ Ensured all output directories exist.")

# --- 1. Build for Custom Framework ---
print("\n--- Building for Custom Framework ---")
# ... (This is the logic from your previous build_knowledge_base.py)
# ... (It saves to the 'knowledge_stores' directory)
publication_urls = custom_da.load_urls_from_file()
irs_content = {name: content for name, url in publication_urls.items() if (content := custom_da.scrape_publication(name, url))}
if irs_content:
    irs_chunks, irs_faiss_index = custom_retriever.build_faiss_index(irs_content)
    faiss.write_index(irs_faiss_index, "knowledge_stores/irs_faiss_index.bin")
    with open("knowledge_stores/irs_chunks.pkl", "wb") as f: pickle.dump(irs_chunks, f)
    print("✅ Custom IRS Knowledge Base built.")
legal_cases_content = custom_da.extract_text_from_pdfs("source_documents/legal_cases")
if legal_cases_content:
    case_chunks, case_faiss_index = custom_retriever.build_faiss_index(legal_cases_content)
    faiss.write_index(case_faiss_index, "knowledge_stores/cases_faiss_index.bin")
    with open("knowledge_stores/cases_chunks.pkl", "wb") as f: pickle.dump(case_chunks, f)
    print("✅ Custom Legal Cases Knowledge Base built.")

# --- 2. Build for LlamaIndex Framework ---
print("\n--- Building for LlamaIndex Framework ---")
LlamaIndex_builder.build_irs_index()
LlamaIndex_builder.build_cases_index()

print("\n✨ Unified build process complete.")