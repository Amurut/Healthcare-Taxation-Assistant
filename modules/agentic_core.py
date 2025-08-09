# modules/agentic_core.py
from modules import llm_clients, retriever
from googlesearch import search
import streamlit as st

def query_llm(messages, llm_choice, api_key, max_tokens=2048):
    """
    Sends a request to the selected LLM, handling different OpenAI model IDs.
    """
    try:
        if llm_choice == "Llama 3 (via Hugging Face)":
            client = llm_clients.get_huggingface_client(api_key)
            if not client: return "Hugging Face API key is missing or invalid."
            response = client.chat_completion(messages=messages, max_tokens=max_tokens, stream=False)
            return response.choices[0].message.content
        
        elif "OpenAI" in llm_choice:
            client = llm_clients.get_openai_client(api_key)
            if not client: return "OpenAI API key is missing or invalid."
            
            # MODIFIED: Use a map to select the correct model ID
            model_map = {
                "OpenAI (GPT-4o)": "gpt-4o",
                "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
                "OpenAI (GPT-4.1-mini)": "gpt-4.1-mini"
            }
            model_id = model_map.get(llm_choice, "gpt-4o-mini") # Default to mini if not found

            response = client.chat.completions.create(model=model_id, messages=messages, max_tokens=max_tokens)
            return response.choices[0].message.content
        
        else:
            return "Error: Invalid LLM choice provided."

    except Exception as e:
        return f"API Error for {llm_choice}: {e}"


def use_irs_knowledge_base(query, chunks, index):
    # ... (this function remains the same)
    return retriever.retrieve_context(query, chunks, index)

def use_legal_cases_knowledge_base(query, chunks, index):
    # ... (this function remains the same)
    return retriever.retrieve_context(query, chunks, index)

def use_web_search(query, num_results=5):
    # ... (this function remains the same)
    try:
        search_results = search(query, num_results=num_results, sleep_interval=1, lang="en")
        links = [result for result in search_results]
        formatted_results = "\n".join([f"- {link}" for link in links])
        return f"Found the following links:\n{formatted_results}", links
    except Exception as e:
        return f"Web search failed: {e}", []

def run_healthcare_tax_agent(main_query, knowledge_bases, llm_choice, api_key):
    # ... (this function remains the same)
    irs_chunks, irs_index = knowledge_bases['irs']
    cases_chunks, cases_index = knowledge_bases['cases']

    plan_prompt = [{"role": "system", "content": "You are a master planner for a legal research agent. Based on the user's query, create a step-by-step plan. The plan should have three parts: 1. Find the IRS rule. 2. Find legal precedents. 3. Find external opinions. For each part, formulate a precise, self-contained query for a search tool."}, {"role": "user", "content": f"User Query: {main_query}"}]
    plan_str = query_llm(plan_prompt, llm_choice, api_key, max_tokens=512)
    plan = [line for line in plan_str.split('\n') if line.strip()]

    irs_query = plan[0] if len(plan) > 0 else main_query
    irs_context, irs_sources = use_irs_knowledge_base(irs_query, irs_chunks, irs_index)
    irs_answer_prompt = [{"role": "system", "content": "Based *only* on the provided IRS context, answer the user's query about the rule. Cite your sources."}, {"role": "user", "content": f"Context:\n{irs_context}\n\nQuery: {irs_query}"}]
    irs_answer = query_llm(irs_answer_prompt, llm_choice, api_key)
    
    cases_query = plan[1] if len(plan) > 1 else main_query
    cases_context, cases_sources = use_legal_cases_knowledge_base(cases_query, cases_chunks, cases_index)
    cases_answer_prompt = [{"role": "system", "content": "Based *only* on the provided legal case context, summarize any relevant precedents related to the user's query. Cite your sources."}, {"role": "user", "content": f"Context:\n{cases_context}\n\nQuery: {cases_query}"}]
    cases_answer = query_llm(cases_answer_prompt, llm_choice, api_key)

    web_query = plan[2] if len(plan) > 2 else f"expert opinions and analysis on healthcare taxation for: {main_query}"
    web_answer, web_sources = use_web_search(web_query)

    return {"plan": plan_str, "irs_answer": irs_answer, "irs_sources": irs_sources, "cases_answer": cases_answer, "cases_sources": cases_sources, "web_search_answer": web_answer}