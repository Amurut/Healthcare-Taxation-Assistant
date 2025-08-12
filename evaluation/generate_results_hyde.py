# evaluation/generate_results_hyde.py
import pandas as pd
import sys
import os
import time
import faiss
import pickle

# Add the project root to the Python path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import agentic_core as custom_agent
from llama_index_modules import LlamaIndex_agent

# --- CONFIGURATION ---
# Ensure your OPENAI_API_KEY is set as an environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

LLM_CHOICE = "OpenAI (GPT-4o)" # The model used for generation

# --- Helper function to load custom KBs ---
def load_custom_kbs():
    knowledge_bases = {'irs': (None, None), 'cases': (None, None)}
    if os.path.exists("knowledge_stores/irs_faiss_index.bin"):
        irs_index = faiss.read_index("knowledge_stores/irs_faiss_index.bin")
        with open("knowledge_stores/irs_chunks.pkl", "rb") as f:
            irs_chunks = pickle.load(f)
        knowledge_bases['irs'] = (irs_chunks, irs_index)
    return knowledge_bases

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # Load the golden dataset
    try:
        eval_df = pd.read_csv("evaluation/eval_dataset.csv")
    except FileNotFoundError:
        print("Error: `evaluation/eval_dataset.csv` not found. Please create it first.")
        sys.exit(1)

    # Load knowledge bases once
    print("Loading knowledge bases for evaluation...")
    custom_kbs = load_custom_kbs()
    llama_indexes = LlamaIndex_agent.load_llama_index_kbs()
    
    all_results = []
    
    for index, row in eval_df.iterrows():
        question = row['question']
        ground_truth = row['ground_truth']
        print(f"\nProcessing question ({index+1}/{len(eval_df)}): {question}")

        # --- Run Custom Code with HyDE ---
        print("  - Running: Custom Code with HyDE")
        try:
            custom_answer, custom_contexts = custom_agent.generate_answer_for_eval_hyde(
                question, custom_kbs, LLM_CHOICE, API_KEY
            )
            all_results.append({
                "framework": "Custom Code",
                "strategy": "HyDE",
                "question": question,
                "ground_truth": ground_truth,
                "answer": custom_answer,
                "contexts": custom_contexts
            })
        except Exception as e:
            print(f"    ERROR in Custom Code: {e}")
        time.sleep(2) # To avoid rate limiting

        # --- Run LlamaIndex with HyDE ---
        print("  - Running: LlamaIndex with HyDE")
        try:
            li_answer, li_contexts = LlamaIndex_agent.generate_answer_for_eval_hyde(
                question, LLM_CHOICE, API_KEY, llama_indexes
            )
            all_results.append({
                "framework": "LlamaIndex",
                "strategy": "HyDE",
                "question": question,
                "ground_truth": ground_truth,
                "answer": li_answer,
                "contexts": li_contexts
            })
        except Exception as e:
            print(f"    ERROR in LlamaIndex: {e}")
        time.sleep(2)

    # Save the results to a new CSV for evaluation
    results_df = pd.DataFrame(all_results)
    output_path = "evaluation/generated_results_hyde.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ HyDE generation complete. Results saved to {output_path}")