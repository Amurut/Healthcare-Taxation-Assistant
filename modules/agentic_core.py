# modules/agentic_core.py
from modules import llm_clients, retriever
from googlesearch import search
import streamlit as st

def query_llm(messages, llm_choice, api_key, max_tokens=2048):
    # ... (this function remains the same as before)
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
        else:
            return "Error: Invalid LLM choice provided."
    except Exception as e:
        return f"API Error for {llm_choice}: {e}"

# --- Agent Tools (no changes to the tool functions themselves) ---
def use_irs_knowledge_base(query, chunks, index):
    return retriever.retrieve_context(query, chunks, index, top_k=5) # Retrieve more context for direct answers

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

# --- NEW: Streamlined Workflow for Direct Answers ---
def run_direct_rag_answer(main_query, knowledge_bases, llm_choice, api_key):
    """
    A simple, direct RAG workflow that only uses the IRS knowledge base
    to answer the user's question directly.
    """
    irs_chunks, irs_index = knowledge_bases['irs']
    
    # 1. Retrieve context from the primary knowledge base (IRS docs)
    context, sources = use_irs_knowledge_base(main_query, irs_chunks, irs_index)
    
    # 2. Generate a direct answer based on the context
    direct_answer_prompt = [{
        "role": "system",
        "content": "You are a precise financial assistant. Answer the user's question directly and concisely based *only* on the provided context from IRS publications. Extract specific numbers, limits, and rules when available. Cite the source publication(s)."
    }, {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {main_query}"
    }]
    
    answer = query_llm(direct_answer_prompt, llm_choice, api_key)
    
    return {
        "answer": answer,
        "sources": sources
    }

# --- EXISTING: Full Agentic Workflow ---
def run_healthcare_tax_agent(main_query, knowledge_bases, llm_choice, api_key):
    """
    The full multi-tool, multi-step agentic workflow.
    """
    # First, get a direct answer from the IRS docs
    direct_answer_result = run_direct_rag_answer(main_query, knowledge_bases, llm_choice, api_key)
    
    # Unpack other knowledge bases
    cases_chunks, cases_index = knowledge_bases['cases']
    
    # Agent Planning Step
    plan_prompt = [{"role": "system", "content": "You are a master planner. Based on the user's query, create a two-step plan: 1. Find legal precedents for the query. 2. Find external opinions for the query. For each part, formulate a precise search query."}, {"role": "user", "content": f"User Query: {main_query}"}]
    plan_str = query_llm(plan_prompt, llm_choice, api_key, max_tokens=512)
    plan = [line for line in plan_str.split('\n') if line.strip()]

    # Execute the rest of the plan
    cases_query = plan[0] if len(plan) > 0 else main_query
    cases_context, cases_sources = use_legal_cases_knowledge_base(cases_query, cases_chunks, cases_index)
    cases_answer_prompt = [{"role": "system", "content": "Based *only* on the provided legal case context, summarize any relevant precedents. Cite sources."}, {"role": "user", "content": f"Context:\n{cases_context}\n\nQuery: {cases_query}"}]
    cases_answer = query_llm(cases_answer_prompt, llm_choice, api_key)

    web_query = plan[1] if len(plan) > 1 else f"expert opinions and analysis on healthcare taxation for: {main_query}"
    web_answer, web_sources = use_web_search(web_query)

    return {
        "direct_answer": direct_answer_result['answer'],
        "direct_sources": direct_answer_result['sources'],
        "plan": plan_str,
        "cases_answer": cases_answer,
        "cases_sources": cases_sources,
        "web_search_answer": web_answer
    }