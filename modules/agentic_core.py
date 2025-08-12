# modules/agentic_core.py
from modules import llm_clients, retriever, query_transformations
from googlesearch import search

def query_llm(messages, llm_choice, api_key, max_tokens=2048):
    """Handles routing to the correct LLM API."""
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

# --- Agent Tools ---
def use_irs_knowledge_base(query, chunks, index):
    return retriever.retrieve_context(query, chunks, index, top_k=5)

def use_legal_cases_knowledge_base(query, chunks, index):
    return retriever.retrieve_context(query, chunks, index)

def use_web_search(query, num_results=5):
    try:
        search_results = search(query, num_results=num_results, sleep_interval=1, lang="en")
        links = [result for result in search_results]
        return f"Found the following links:\n" + "\n".join([f"- {link}" for link in links]), links
    except Exception as e:
        return f"Web search failed: {e}", []

# --- Main Workflows ---
def run_direct_rag_answer(main_query, knowledge_bases, llm_choice, api_key, retrieval_strategy="Standard"):
    """Handles retrieval strategy internally and runs the self-correction loop."""
    irs_chunks, irs_index = knowledge_bases['irs']
    
    retrieved_context, sources, strategy_details = "", [], {}
    
    if retrieval_strategy == "HyDE":
        retrieved_context, sources, hypo_doc = query_transformations.retrieve_with_hyde(main_query, llm_choice, api_key, irs_chunks, irs_index)
        strategy_details = {"title": "HyDE: Hypothetical Document", "content": hypo_doc}
    elif retrieval_strategy == "Multi-Query":
        retrieved_context, sources, sub_queries = query_transformations.retrieve_with_multi_query(main_query, llm_choice, api_key, irs_chunks, irs_index)
        strategy_details = {"title": "Multi-Query: Generated Sub-Queries", "content": "\n- ".join([""] + sub_queries)}
    else: # Standard
        retrieved_context, sources = retriever.retrieve_context(main_query, irs_chunks, irs_index)

    gen_prompt = [{"role": "system", "content": "You are a precise financial assistant. Based *only* on the provided context, provide a direct and crisp answer to the user's question. Extract specific numbers, limits, and rules when available."}, {"role": "user", "content": f"Context:\n{retrieved_context}\n\nQuestion: {main_query}"}]
    initial_answer = query_llm(gen_prompt, llm_choice, api_key)

    critique_prompt = [{"role": "system", "content": "You are a fact-checker. Critique the 'Draft Answer'. Is it faithful and direct? Suggest improvements."}, {"role": "user", "content": f"Context:\n{retrieved_context}\n\nDraft Answer:\n{initial_answer}"}]
    critique = query_llm(critique_prompt, llm_choice, api_key, max_tokens=512)

    refine_prompt = [{"role": "system", "content": "You are a financial assistant. Refine the 'Draft Answer' using the 'Critique' to create a final, improved response. Cite the source publication(s)."}, {"role": "user", "content": f"User's Original Question: {main_query}\n\nContext:\n{retrieved_context}\n\nDraft Answer:\n{initial_answer}\n\nCritique:\n{critique}\n\nFinal Improved Answer:"}]
    final_answer = query_llm(refine_prompt, llm_choice, api_key)
    
    return {
        "initial": initial_answer,
        "critique": critique,
        "final": final_answer,
        "sources": sources,
        "query_transformation": strategy_details
    }

def run_healthcare_tax_agent(main_query, knowledge_bases, llm_choice, api_key, retrieval_strategy="Standard"):
    """The full multi-tool agentic workflow."""
    direct_rag_results = run_direct_rag_answer(main_query, knowledge_bases, llm_choice, api_key, retrieval_strategy)
    
    cases_chunks, cases_index = knowledge_bases['cases']
    
    plan_prompt = [{"role": "system", "content": "Create a two-step plan: 1. Find legal precedents for the query. 2. Find external opinions for the query. Formulate a precise search query for each step."}, {"role": "user", "content": f"User Query: {main_query}"}]
    plan_str = query_llm(plan_prompt, llm_choice, api_key, max_tokens=512)
    plan = [line for line in plan_str.split('\n') if line.strip()]

    cases_query = plan[0] if len(plan) > 0 else main_query
    cases_context, cases_sources = use_legal_cases_knowledge_base(cases_query, cases_chunks, cases_index)
    cases_answer_prompt = [{"role": "system", "content": "Based *only* on the provided legal case context, summarize any relevant precedents. Cite sources."}, {"role": "user", "content": f"Context:\n{cases_context}\n\nQuery: {cases_query}"}]
    cases_answer = query_llm(cases_answer_prompt, llm_choice, api_key)

    web_query = plan[1] if len(plan) > 1 else f"expert opinions and analysis on healthcare taxation for: {main_query}"
    web_answer, web_sources = use_web_search(web_query)

    return {
        "direct_answer_results": direct_rag_results,
        "plan": plan_str,
        "cases_answer": cases_answer,
        "cases_sources": cases_sources,
        "web_search_answer": web_answer
    }