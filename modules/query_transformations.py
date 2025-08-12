from modules import retriever, agentic_core

def retrieve_with_hyde(query, llm_choice, api_key, all_chunks, index, top_k=5):
    """
    Generates a hypothetical document and uses its embedding for retrieval.
    """
    print("Executing retrieval with HyDE...")
    # 1. Generate a hypothetical document
    hyde_prompt = [{
        "role": "system",
        "content": "You are a helpful assistant. The user will ask a question. Generate a concise, hypothetical document that could plausibly answer this question. Do not include any pre-amble."
    }, {
        "role": "user",
        "content": query
    }]
    hypothetical_document = agentic_core.query_llm(hyde_prompt, llm_choice, api_key, max_tokens=512)
    
    # 2. Use the embedding of the hypothetical document for retrieval
    # We reuse the core retrieve_context function but pass the new document as the query
    context, sources = retriever.retrieve_context(hypothetical_document, all_chunks, index, top_k)
    
    # We also pass back the hypothetical document for transparency
    return context, sources, hypothetical_document

def retrieve_with_multi_query(query, llm_choice, api_key, all_chunks, index, top_k=3):
    """
    Generates multiple sub-queries and retrieves documents for all of them.
    """
    print("Executing retrieval with Multi-Query...")
    # 1. Generate multiple sub-queries
    multi_query_prompt = [{
        "role": "system",
        "content": "You are a helpful assistant. The user will ask a complex question. Your task is to generate 3-5 different, simpler sub-queries that capture various facets of the original question. Output each query on a new line, without numbering."
    }, {
        "role": "user",
        "content": query
    }]
    
    sub_queries_str = agentic_core.query_llm(multi_query_prompt, llm_choice, api_key, max_tokens=512)
    sub_queries = [q.strip() for q in sub_queries_str.split('\n') if q.strip()]
    
    # 2. Retrieve documents for each sub-query
    all_retrieved_contexts = []
    all_retrieved_sources = set()
    
    for sub_q in sub_queries:
        context, sources = retriever.retrieve_context(sub_q, all_chunks, index, top_k)
        all_retrieved_contexts.append(context)
        all_retrieved_sources.update(sources)
        
    # 3. Combine and de-duplicate the context
    # A simple way is to just join them. More advanced de-duplication can be added later.
    final_context = "\n\n".join(all_retrieved_contexts)
    
    return final_context, list(all_retrieved_sources), sub_queries