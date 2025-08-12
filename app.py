import streamlit as st
import faiss
import pickle
import os
from openai import OpenAI
from modules import agentic_core as custom_agent, data_acquisition, retriever, llm_clients, query_transformations
from llama_index_modules import LlamaIndex_agent

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Healthcare Taxation Assistant", page_icon="⚕️", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please select a framework, authenticate your API key, and ask a question."}]
if "auth_status" not in st.session_state:
    st.session_state.auth_status = {}
if "framework_choice" not in st.session_state:
    st.session_state.framework_choice = "Custom Code"
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "OpenAI (GPT-4o)"
if "retrieval_strategy" not in st.session_state:
    st.session_state.retrieval_strategy = "Standard"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.selectbox("Choose Framework:", ("Custom Code", "LlamaIndex"), key="framework_choice")
    st.selectbox(
        "Choose Retrieval Strategy:",
        ("Standard", "HyDE", "Multi-Query"),
        key="retrieval_strategy",
        help="Standard: Direct search. HyDE: Creates a hypothetical answer to improve search. Multi-Query: Breaks your question into sub-questions."
    )
    st.selectbox("Choose Language Model:", ("OpenAI (GPT-4o)", "OpenAI (GPT-4o-mini)", "OpenAI (GPT-4.1-mini)"), key="llm_choice")
    with st.form("api_key_form"):
        api_key_input = st.text_input(f"Enter {st.session_state.llm_choice} API Key", type="password")
        submitted = st.form_submit_button("Submit & Authenticate Key")
        if submitted:
            with st.spinner("Authenticating..."):
                is_valid, message = llm_clients.verify_api_key(st.session_state.llm_choice, api_key_input)
                if is_valid:
                    st.session_state.auth_status[st.session_state.llm_choice] = api_key_input
                    st.success(message)
                else:
                    st.session_state.auth_status[st.session_state.llm_choice] = None
                    st.error(message)
    if st.session_state.auth_status.get(st.session_state.llm_choice):
        st.success(f"✅ Key for {st.session_state.llm_choice} is active.")
    else:
        st.warning(f"⚠️ Key for {st.session_state.llm_choice} is not set or invalid.")

# --- KNOWLEDGE BASE LOADING ---
@st.cache_resource(show_spinner="Initializing Custom Knowledge Bases...")
def load_custom_kbs():
    knowledge_bases = {'irs': (None, None), 'cases': (None, None)}
    if os.path.exists("knowledge_stores/irs_faiss_index.bin"):
        irs_index = faiss.read_index("knowledge_stores/irs_faiss_index.bin")
        with open("knowledge_stores/irs_chunks.pkl", "rb") as f: irs_chunks = pickle.load(f)
        knowledge_bases['irs'] = (irs_chunks, irs_index)
    if os.path.exists("knowledge_stores/cases_faiss_index.bin"):
        cases_index = faiss.read_index("knowledge_stores/cases_faiss_index.bin")
        with open("knowledge_stores/cases_chunks.pkl", "rb") as f: cases_chunks = pickle.load(f)
        knowledge_bases['cases'] = (cases_chunks, cases_index)
    return knowledge_bases

# --- MAIN APP INTERFACE ---
st.title("⚕️ Healthcare Taxation Assistant")
st.caption(f"Using: **{st.session_state.framework_choice}** | Strategy: **{st.session_state.retrieval_strategy}** | Model: **{st.session_state.llm_choice}**")

if st.session_state.framework_choice == "Custom Code":
    knowledge_bases = load_custom_kbs()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))
        if msg["role"] == "user" and msg.get("label"):
            st.caption(msg["label"])
        
        if msg["role"] == "assistant":
            if msg.get("query_transformation") and msg["query_transformation"].get("content"):
                with st.expander("Show Query Transformation"):
                    st.subheader(msg["query_transformation"]["title"])
                    st.info(msg["query_transformation"]["content"])
            if msg.get("thought_process"):
                with st.expander("Show Self-Correction Process"):
                    st.info(f"**Sources:** {', '.join(msg['thought_process']['sources'])}")
                    st.warning(f"**Initial Draft:**\n{msg['thought_process']['initial']}")
                    st.error(f"**Critique:**\n{msg['thought_process']['critique']}")
            if msg.get("full_analysis"):
                with st.expander("Show Full Agentic Analysis"):
                    st.code(f"Agent's Plan:\n{msg['full_analysis']['plan']}", language="text")
                    st.success(f"**Legal & Web Analysis:**\n{msg['full_analysis']['agent_response']}")

# Handles chat input and response generation
if prompt := st.chat_input("Ask about healthcare tax rules..."):
    # First, save and display the user's prompt, then rerun to show it immediately
    current_label = f"Framework: {st.session_state.framework_choice} | Strategy: {st.session_state.retrieval_strategy} | Model: {st.session_state.llm_choice}"
    st.session_state.messages.append({"role": "user", "content": prompt, "label": current_label})
    st.rerun()

# This block runs AFTER the rerun, ensuring the user's prompt is already on screen
if st.session_state.messages[-1]["role"] == "user":
    active_api_key = st.session_state.auth_status.get(st.session_state.llm_choice)
    if not active_api_key:
        st.info("Please submit a valid API key in the sidebar.")
        st.stop()
    
    prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        message_to_save = {"role": "assistant"}
        final_response = ""

        if 'show legal precedent' in prompt.lower():
            if st.session_state.framework_choice == "Custom Code":
                with st.spinner("Custom agent is performing full analysis..."):
                    results = custom_agent.run_healthcare_tax_agent(prompt, knowledge_bases, st.session_state.llm_choice, active_api_key, st.session_state.retrieval_strategy)
                    direct_results = results['direct_answer_results']
                    final_response = f"**Direct Answer from IRS Rules:**\n{direct_results['final']}"
                    if direct_results.get("query_transformation"):
                        message_to_save["query_transformation"] = direct_results["query_transformation"]
                    message_to_save["thought_process"] = direct_results
                    message_to_save["full_analysis"] = {"plan": results['plan'], "agent_response": f"{results['cases_answer']}\n{results['web_search_answer']}"}
            else: # LlamaIndex
                with st.spinner("LlamaIndex agent is performing full analysis..."):
                    direct_results, agent_response = LlamaIndex_agent.run_llama_index_agent(prompt, st.session_state.llm_choice, active_api_key, st.session_state.retrieval_strategy)
                    final_response = f"**Direct Answer from IRS Rules:**\n{direct_results['final']}"
                    if direct_results.get("query_transformation"):
                        message_to_save["query_transformation"] = direct_results["query_transformation"]
                    message_to_save["thought_process"] = direct_results
                    message_to_save["full_analysis"] = {"plan": "Executed via LlamaIndex ReAct Agent", "agent_response": agent_response}
        else: # Default direct answer
            if st.session_state.framework_choice == "Custom Code":
                with st.spinner(f"Custom agent using '{st.session_state.retrieval_strategy}'..."):
                    results = custom_agent.run_direct_rag_answer(prompt, knowledge_bases, st.session_state.llm_choice, active_api_key, st.session_state.retrieval_strategy)
                    final_response = results['final']
                    message_to_save["thought_process"] = results
                    if results.get("query_transformation"):
                        message_to_save["query_transformation"] = results["query_transformation"]
            else: # LlamaIndex
                with st.spinner(f"LlamaIndex using '{st.session_state.retrieval_strategy}'..."):
                    indexes = LlamaIndex_agent.load_llama_index_kbs()
                    results = LlamaIndex_agent.run_direct_llama_index_query(prompt, st.session_state.llm_choice, active_api_key, indexes, st.session_state.retrieval_strategy)
                    final_response = results['final']
                    message_to_save["thought_process"] = results
                    if results.get("query_transformation"):
                        message_to_save["query_transformation"] = results["query_transformation"]
        
        # Display the new response and its expanders
        st.markdown(final_response)
        if "query_transformation" in message_to_save and message_to_save["query_transformation"].get("content"):
            with st.expander("Show Query Transformation"):
                st.subheader(message_to_save["query_transformation"]["title"])
                st.info(message_to_save["query_transformation"]["content"])
        if "thought_process" in message_to_save:
            with st.expander("Show Self-Correction Process"):
                st.info(f"**Sources:** {', '.join(message_to_save['thought_process']['sources'])}")
                st.warning(f"**Initial Draft:**\n{message_to_save['thought_process']['initial']}")
                st.error(f"**Critique:**\n{message_to_save['thought_process']['critique']}")
        if "full_analysis" in message_to_save:
            with st.expander("Show Full Agentic Analysis"):
                st.code(f"Agent's Plan:\n{message_to_save['full_analysis']['plan']}", language="text")
                st.success(f"**Legal & Web Analysis:**\n{message_to_save['full_analysis']['agent_response']}")
        
        # Save the complete assistant message to history
        message_to_save["content"] = final_response
        st.session_state.messages.append(message_to_save)