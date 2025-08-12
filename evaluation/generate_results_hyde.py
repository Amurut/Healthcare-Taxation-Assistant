import pandas as pd
import sys
import os
import time
import faiss
import pickle
import streamlit as st # Import Streamlit for st.cache_resource

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import agentic_core as custom_agent, query_transformations
from llama_index_modules import LlamaIndex_agent

# --- CONFIGURATION ---
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

LLM_CHOICE = "OpenAI (GPT-4o)"

# --- Helper function to load custom KBs ---
@st.cache_resource
def load_custom_kbs():
    knowledge_bases = {'irs': (None, None)}
    if os.path.exists("knowledge_stores/irs_faiss_index.bin"):
        irs_index = faiss.read_index("knowledge_stores/irs_faiss_index.bin")
        with open("knowledge_stores/irs_chunks.pkl", "rb") as f:
            irs_chunks = pickle.load(f)
        knowledge_bases['irs'] = (irs_chunks, irs_index)
    return knowledge_bases

# --- New simplified generation functions for evaluation ---
def generate_custom_answer(query, kbs):
    context, sources, _ = query_transformations.retrieve_with_hyde(
        query, LLM_CHOICE, API_KEY, kbs['irs'][0], kbs['irs'][1]
    )
    prompt = [{"role": "system", "content": "Answer the question based only on the context."}, 
              {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    answer = custom_agent.query_llm(prompt, LLM_CHOICE, API_KEY)
    return answer, [c.strip() for c in context.split("---") if c.strip()]

def generate_llama_index_answer(query, indexes):
    response, context_list = LlamaIndex_agent.generate_answer_for_eval_hyde(
        query, LLM_CHOICE, API_KEY, indexes
    )
    return response, context_list

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    eval_df = pd.read_csv("evaluation/eval_dataset.csv")
    
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
            custom_answer, custom_contexts = generate_custom_answer(question, custom_kbs)
            all_results.append({
                "framework": "Custom Code", "strategy": "HyDE", "question": question,
                "ground_truth": ground_truth, "answer": custom_answer, "contexts": custom_contexts
            })
        except Exception as e:
            print(f"    ERROR in Custom Code: {e}")
        time.sleep(1)

        # --- Run LlamaIndex with HyDE ---
        print("  - Running: LlamaIndex with HyDE")
        try:
            li_answer, li_contexts = generate_llama_index_answer(question, llama_indexes)
            all_results.append({
                "framework": "LlamaIndex", "strategy": "HyDE", "question": question,
                "ground_truth": ground_truth, "answer": li_answer, "contexts": li_contexts
            })
        except Exception as e:
            print(f"    ERROR in LlamaIndex: {e}")
        time.sleep(1)

    results_df = pd.DataFrame(all_results)
    output_path = "evaluation/generated_results_hyde.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ HyDE generation complete. Results saved to {output_path}")