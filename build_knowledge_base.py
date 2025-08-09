# build_knowledge_base.py
import pickle
import faiss
import os
from modules import data_acquisition, retriever

print("🚀 Starting Knowledge Base build process...")

# --- Automatically create output directories if they don't exist ---
output_dir = "knowledge_stores"
debug_dir = "debug_outputs" # Directory for text dumps
os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True) # Create debug directory
print(f"✅ Ensured output directories '{output_dir}' and '{debug_dir}' exist.")

# --- Build IRS Knowledge Base ---
print("\n[1/2] Building IRS Publications Knowledge Base...")
publication_urls = data_acquisition.load_urls_from_file()
irs_content = {}
for name, url in publication_urls.items():
    print(f"  -> Scraping: {name} ({url})")
    content = data_acquisition.scrape_publication(name, url)
    if content:
        irs_content[name] = content

if irs_content:
    # --- DEV-ONLY: Write scraped web content to a debug file ---
    web_content_path = os.path.join(debug_dir, "scraped_web_content.txt")
    print(f"  -> Writing scraped content to '{web_content_path}' for debugging...")
    with open(web_content_path, "w", encoding="utf-8") as f:
        for source, text in irs_content.items():
            f.write(f"\n{'='*20} START OF: {source} {'='*20}\n\n")
            f.write(text)
            f.write(f"\n\n{'='*20} END OF: {source} {'='*20}\n")
    # -----------------------------------------------------------

    irs_chunks, irs_faiss_index = retriever.build_faiss_index(irs_content)
    faiss.write_index(irs_faiss_index, os.path.join(output_dir, "irs_faiss_index.bin"))
    with open(os.path.join(output_dir, "irs_chunks.pkl"), "wb") as f:
        pickle.dump(irs_chunks, f)
    print("✅ IRS Knowledge Base built and saved.")
else:
    print("⚠️ No IRS content scraped. Skipping IRS knowledge base build.")

# --- Build Legal Cases Knowledge Base ---
print("\n[2/2] Building Legal Cases Knowledge Base...")
pdf_folder = "source_documents/legal_cases"
legal_cases_content = data_acquisition.extract_text_from_pdfs(pdf_folder)

if legal_cases_content:
    print(f"  -> Found and processed {len(legal_cases_content)} PDF(s) from '{pdf_folder}':")
    for filename in legal_cases_content.keys():
        print(f"     - {filename}")
        
    # --- DEV-ONLY: Write extracted PDF content to a debug file ---
    pdf_content_path = os.path.join(debug_dir, "extracted_pdf_content.txt")
    print(f"  -> Writing extracted PDF content to '{pdf_content_path}' for debugging...")
    with open(pdf_content_path, "w", encoding="utf-8") as f:
        for source, text in legal_cases_content.items():
            f.write(f"\n{'='*20} START OF: {source} {'='*20}\n\n")
            f.write(text)
            f.write(f"\n\n{'='*20} END OF: {source} {'='*20}\n")
    # ---------------------------------------------------------

    case_chunks, case_faiss_index = retriever.build_faiss_index(legal_cases_content)
    faiss.write_index(case_faiss_index, os.path.join(output_dir, "cases_faiss_index.bin"))
    with open(os.path.join(output_dir, "cases_chunks.pkl"), "wb") as f:
        pickle.dump(case_chunks, f)
    print("✅ Legal Cases Knowledge Base built and saved.")
else:
    print(f"⚠️ No legal case PDFs found or processed in '{pdf_folder}'. Skipping build.")

print("\n✨ Build process complete.")