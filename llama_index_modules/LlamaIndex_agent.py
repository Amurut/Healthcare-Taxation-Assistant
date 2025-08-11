import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.openai import OpenAI
# Corrected Import: ReActAgent is now in llama_index.core.agent
from llama_index.core.agent import ReActAgent
from googlesearch import search
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Global Settings Configuration ---
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# --- Agent Tools ---
def web_search_tool(query: str) -> str:
    """Performs a web search for opinions and external analyses on a topic."""
    try:
        search_results = search(query, num_results=5, sleep_interval=1, lang="en")
        links = [result for result in search_results]
        return f"Found the following links:\n" + "\n".join(links)
    except Exception as e:
        return f"Web search failed: {e}"

# --- Knowledge Base Loading ---
@st.cache_resource(show_spinner="Loading LlamaIndex knowledge bases...")
def load_llama_index_kbs():
    """Loads the pre-built LlamaIndex vector stores from disk."""
    try:
        irs_storage_context = StorageContext.from_defaults(persist_dir="llama_index_stores/irs_index")
        irs_index = load_index_from_storage(irs_storage_context)

        cases_storage_context = StorageContext.from_defaults(persist_dir="llama_index_stores/cases_index")
        cases_index = load_index_from_storage(cases_storage_context)
        
        st.sidebar.success("LlamaIndex KBs loaded.")
        return {"irs": irs_index, "cases": cases_index}
    except FileNotFoundError:
        st.error("LlamaIndex stores not found. Please run `build_all_kbs.py`.")
        return None

# --- Main Functions for App ---
def run_direct_llama_index_query(query, llm_choice, api_key, indexes):
    """Performs a direct, self-correcting query against the IRS index."""
    model_map = {
        "OpenAI (GPT-4o)": "gpt-4o",
        "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
        "OpenAI (GPT-4.1-mini)": "gpt-4.1-mini"
    }
    model_id = model_map.get(llm_choice, "gpt-4o-mini")
    llm = OpenAI(model=model_id, api_key=api_key)
    Settings.llm = llm

    irs_engine = indexes["irs"].as_query_engine(similarity_top_k=5)
    initial_response = irs_engine.query(query)
    
    context_for_critique = ''.join([node.get_content() for node in initial_response.source_nodes])
    critique_prompt = f"Critique this answer based ONLY on the provided context. Is it accurate and direct?\n\nContext:\n{context_for_critique}\n\nAnswer:\n{initial_response}"
    correction_response = llm.complete(critique_prompt)

    final_prompt = f"Refine the 'Original Answer' using the 'Critique' to provide a final, direct response. Cite sources.\n\nUser Question: {query}\nContext:\n{context_for_critique}\n\nOriginal Answer: {initial_response}\nCritique: {correction_response}\n\nFinal Answer:"
    final_response = llm.complete(final_prompt)
    
    sources = [node.metadata.get('file_name', 'IRS Publication') for node in initial_response.source_nodes]
    
    return {
        "initial": str(initial_response),
        "critique": str(correction_response),
        "final": str(final_response),
        "sources": list(set(sources))
    }

def run_llama_index_agent(query, llm_choice, api_key):
    """Initializes and runs the full LlamaIndex ReActAgent for deep analysis."""
    indexes = load_llama_index_kbs()
    if not indexes:
        return "Could not load LlamaIndex KBs.", {}

    model_map = {
        "OpenAI (GPT-4o)": "gpt-4o",
        "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
        "OpenAI (GPT-4.1-mini)": "gpt-4.1-mini"
    }
    model_id = model_map.get(llm_choice, "gpt-4o-mini")
    llm = OpenAI(model=model_id, api_key=api_key)
    Settings.llm = llm

    direct_answer_results = run_direct_llama_index_query(query, llm_choice, api_key, indexes)

    cases_engine = indexes["cases"].as_query_engine(similarity_top_k=3)
    cases_tool = QueryEngineTool.from_defaults(query_engine=cases_engine, name="legal_precedent_search", description="Search legal case documents for relevant precedents.")
    web_tool = FunctionTool.from_defaults(fn=web_search_tool, name="web_search", description="Search the web for external opinions and analyses.")

    agent = ReActAgent.from_tools(tools=[cases_tool, web_tool], llm=llm, verbose=True)
    agent_task = f"First, find legal precedents for '{query}'. Second, find external opinions for '{query}'. Synthesize the results from these two tasks."
    agent_response = agent.chat(agent_task)

    return direct_answer_results, str(agent_response)