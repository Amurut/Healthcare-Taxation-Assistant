# modules/agentic_core.py
from modules import llm_clients, retriever
from googlesearch import search
import streamlit as st

def query_llm(messages, llm_choice, api_key, max_tokens=2048):
    # ... (this function remains the same)
    try:
        if "OpenAI" in llm_choice:
            client = llm_clients.get_openai_client(api_key)
            if not client: return "OpenAI API key is missing or invalid."
            model_map = {"OpenAI (GPT-4o)": "gpt-4o", "OpenAI (GPT-4o-mini)": "gpt-4o-mini", "OpenAI (GPT-4.1-mini)": "gpt-4.1-mini"}
            model_id = model_map.get(llm_choice, "gpt-4o-mini")
            response = client.chat.completions.create(model=model_id, messages=messages, max_tokens=max_tokens)
            return response.choices[0].message.content
        elif "Llama 3" in llm_choice:
            client = llm_clients.get_huggingface_client(api_key)
            if not client: return "Hugging Face API key is missing or invalid."
            response = client.chat_completion(messages=messages, max_tokens=max_tokens, stream=False)
            return response.choices[0].message.content
    except Exception as e:
        return f"API Error for {llm_choice}: {e}"

# --- Agent Tools (no changes to these functions) ---
def use_irs_knowledge_base(query, chunks, index):
    return retriever.retrieve_context(query, chunks, index, top_k=5)

def use_legal_cases_knowledge_base(query, chunks, index):
    return retriever.retrieve_context(query, chunks, index)

def use_web_search(query, num_results=5):
    try:
        search_results = search(query, num_results=num_results, sleep_interval=1, lang="en")
        links = [result for result in search_results]
        formatted_results = "\n".join([f"- {link}" for link in links])
        return f"Found the following links:\n{formatted_results}", links
    except Exception as e:
        return f"Web search failed: {e}", []

# --- UPDATED: Direct RAG Workflow with Self-Correction ---
def run_direct_rag_answer(main_query, knowledge_bases, llm_choice, api_key):
    """
    A direct RAG workflow that uses a self-correction loop on the IRS knowledge base.
    """
    irs_chunks, irs_index = knowledge_bases['irs']
    context, sources = use_irs_knowledge_base(main_query, irs_chunks, irs_index)

    # 1. Generate Initial Answer
    gen_prompt = [{"role": "system", "content": "You are a financial assistant. Based *only* on the provided context, give a direct answer to the user's question. Extract specific numbers if available."}, {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {main_query}"}]
    initial_answer = query_llm(gen_prompt, llm_choice, api_key)

    # 2. Critique the Answer
    critique_prompt = [{"role": "system", "content": "You are a meticulous fact-checker. Critique the 'Draft Answer'. Did it correctly extract specific numbers and rules from the context? Is it faithful to the source? Suggest improvements for accuracy and directness."}, {"role": "user", "content": f"Context:\n{context}\n\nDraft Answer:\n{initial_answer}"}]
    critique = query_llm(critique_prompt, llm_choice, api_key, max_tokens=512)

    # 3. Refine the Answer
    refine_prompt = [{"role": "system", "content": "You are a financial assistant. Refine the 'Draft Answer' using the 'Critique' to create a final, improved response that directly answers the user's original question using only the provided context. Cite the source publication(s)."}, {"role": "user", "content": f"User's Original Question: {main_query}\n\nContext:\n{context}\n\nDraft Answer:\n{initial_answer}\n\nCritique:\n{critique}\n\nFinal Improved Answer:"}]
    final_answer = query_llm(refine_prompt, llm_choice, api_key)
    
    return {
        "initial": initial_answer,
        "critique": critique,
        "final": final_answer,
        "sources": sources
    }

# --- Full Agentic Workflow ---
def run_healthcare_tax_agent(main_query, knowledge_bases, llm_choice, api_key):
    """
    The full multi-tool, multi-step agentic workflow.
    """
    # This now starts with the self-correcting direct answer
    direct_rag_results = run_direct_rag_answer(main_query, knowledge_bases, llm_choice, api_key)
    
    cases_chunks, cases_index = knowledge_bases['cases']
    
    plan_prompt = [{"role": "system", "content": "You are a master planner. Create a two-step plan: 1. Find legal precedents for the query. 2. Find external opinions for the query. For each part, formulate a precise search query."}, {"role": "user", "content": f"User Query: {main_query}"}]
    plan_str = query_llm(plan_prompt, llm_choice, api_key, max_tokens=512)
    plan = [line for line in plan_str.split('\n') if line.strip()]

    cases_query = plan[0] if len(plan) > 0 else main_query
    cases_context, cases_sources = use_legal_cases_knowledge_base(cases_query, cases_chunks, cases_index)
    cases_answer_prompt = [{"role": "system", "content": "Based *only* on the provided legal case context, summarize any relevant precedents. Cite sources."}, {"role": "user", "content": f"Context:\n{cases_context}\n\nQuery: {cases_query}"}]
    cases_answer = query_llm(cases_answer_prompt, llm_choice, api_key)

    web_query = plan[1] if len(plan) > 1 else f"expert opinions and analysis on healthcare taxation for: {main_query}"
    web_answer, web_sources = use_web_search(web_query)

    return {
        "direct_answer": direct_rag_results, # Pass the entire result dictionary
        "plan": plan_str,
        "cases_answer": cases_answer,
        "cases_sources": cases_sources,
        "web_search_answer": web_answer
    }