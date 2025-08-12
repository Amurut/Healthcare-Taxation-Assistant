# llama_index_modules/LlamaIndex_agent.py
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from googlesearch import search
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index_modules import query_transformations
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

def web_search_tool(query: str) -> str:
    """Performs a web search for opinions and external analyses on a topic."""
    try:
        search_results = search(query, num_results=5, sleep_interval=1, lang="en")
        return f"Found the following links:\n" + "\n".join([result for result in search_results])
    except Exception as e:
        return f"Web search failed: {e}"

@st.cache_resource(show_spinner="Loading LlamaIndex knowledge bases...")
def load_llama_index_kbs():
    """Loads the pre-built LlamaIndex vector stores from disk."""
    try:
        irs_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="llama_index_stores/irs_index"))
        cases_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="llama_index_stores/cases_index"))
        st.sidebar.success("LlamaIndex KBs loaded.")
        return {"irs": irs_index, "cases": cases_index}
    except FileNotFoundError:
        st.error("LlamaIndex stores not found. Please run `build_all_kbs.py`.")
        return None

def run_direct_llama_index_query(query, llm_choice, api_key, indexes, retrieval_strategy="Standard"):
    """Performs a direct query using the selected retrieval strategy."""
    model_map = {"OpenAI (GPT-4o)": "gpt-4o", "OpenAI (GPT-4o-mini)": "gpt-4o-mini", "OpenAI (GPT-4.1-mini)": "gpt-4.1-mini"}
    model_id = model_map.get(llm_choice, "gpt-4o-mini")
    llm = OpenAI(model=model_id, api_key=api_key)
    
    llama_debug = LlamaDebugHandler(print_trace_on_end=False)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager
    Settings.llm = llm

    irs_engine = indexes["irs"].as_query_engine(similarity_top_k=2)
    strategy_details = {}

    if retrieval_strategy == "HyDE":
        query_engine = query_transformations.get_hyde_query_engine(indexes["irs"], llm)
    elif retrieval_strategy == "Multi-Query":
        query_engine_tool = QueryEngineTool.from_defaults(query_engine=irs_engine, name="irs_rules_search", description="Use for questions about U.S. healthcare taxation and IRS rules.")
        query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=[query_engine_tool], llm=llm, verbose=True)
    else: # Standard
        query_engine = irs_engine
    
    initial_response = query_engine.query(query)
    
    if retrieval_strategy == "Multi-Query":
        sub_questions = []
        for event in llama_debug.get_events():
            if event.event_type == 'llm' and "Sub question" in str(event.payload):
                payload_str = str(event.payload)
                questions_part = payload_str.split("Sub question an lyric:")[-1]
                questions = [q.strip().split(":")[-1].strip() for q in questions_part.split("\\n") if q.strip()]
                sub_questions = questions
                break
        strategy_details = {"title": "LlamaIndex Multi-Query: Generated Sub-Queries", "content": "\n- ".join([""] + sub_questions)}

    context_for_critique = ''.join([node.get_content() for node in initial_response.source_nodes])
    critique_prompt = f"Critique this answer based ONLY on the provided context...\nContext:\n{context_for_critique}\n\nAnswer:\n{initial_response}"
    correction_response = llm.complete(critique_prompt)
    final_prompt = f"Refine the 'Original Answer' using the 'Critique'...\nUser Question: {query}\nContext:\n{context_for_critique}\n\nOriginal Answer: {initial_response}\nCritique: {correction_response}\n\nFinal Answer:"
    final_response = llm.complete(final_prompt)
    
    sources = [node.metadata.get('file_name', 'IRS Publication') for node in initial_response.source_nodes]
    
    return {"initial": str(initial_response), "critique": str(correction_response), "final": str(final_response), "sources": list(set(sources)), "query_transformation": strategy_details}

def run_llama_index_agent(query, llm_choice, api_key, retrieval_strategy="Standard"):
    """Initializes and runs the full LlamaIndex ReActAgent for deep analysis."""
    indexes = load_llama_index_kbs()
    if not indexes:
        return "Could not load LlamaIndex KBs.", {}

    model_map = {"OpenAI (GPT-4o)": "gpt-4o", "OpenAI (GPT-4o-mini)": "gpt-4o-mini", "OpenAI (GPT-4.1-mini)": "gpt-4.1-mini"}
    model_id = model_map.get(llm_choice, "gpt-4o-mini")
    llm = OpenAI(model=model_id, api_key=api_key)
    Settings.llm = llm

    direct_answer_results = run_direct_llama_index_query(query, llm_choice, api_key, indexes, retrieval_strategy)

    cases_engine = indexes["cases"].as_query_engine(similarity_top_k=3)
    cases_tool = QueryEngineTool.from_defaults(query_engine=cases_engine, name="legal_precedent_search", description="Search legal case documents for relevant precedents.")
    web_tool = FunctionTool.from_defaults(fn=web_search_tool, name="web_search", description="Search the web for external opinions and analyses.")

    agent = ReActAgent.from_tools(tools=[cases_tool, web_tool], llm=llm, verbose=True)
    agent_task = f"First, find legal precedents for '{query}'. Second, find external opinions for '{query}'. Synthesize the results from these two tasks."
    agent_response = agent.chat(agent_task)

    return direct_answer_results, str(agent_response)


def generate_answer_for_eval_hyde(query, llm_choice, api_key, indexes):
    """
    A simplified one-shot generation process using LlamaIndex HyDE for evaluation.
    Returns: (answer_string, list_of_context_strings)
    """
    model_map = {"OpenAI (GPT-4o)": "gpt-4o", "OpenAI (GPT-4o-mini)": "gpt-4o-mini", "OpenAI (GPT-4.1-mini)": "gpt-4.1-mini"}
    model_id = model_map.get(llm_choice, "gpt-4o-mini")
    llm = OpenAI(model=model_id, api_key=api_key)
    Settings.llm = llm

    # Get the HyDE query engine
    query_engine = query_transformations.get_hyde_query_engine(indexes["irs"], llm)
    
    # Generate the response
    response = query_engine.query(query)
    
    # RAGAS expects context as a list of strings
    context_list = [node.get_content() for node in response.source_nodes]
    
    return str(response), context_list