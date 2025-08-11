# app.py
import streamlit as st
import faiss
import pickle
import os
from openai import OpenAI
from modules import agentic_core as custom_agent, data_acquisition, retriever, llm_clients
from llama_index_modules import LlamaIndex_agent

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Healthcare Taxation Assistant", page_icon="⚕️", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please select a framework, authenticate your API key, and ask a question about healthcare taxation."}]
if "auth_status" not in st.session_state:
    st.session_state.auth_status = {}
if "framework_choice" not in st.session_state:
    st.session_state.framework_choice = "Custom Code"
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "OpenAI (GPT-4o)"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.selectbox("Choose Framework:", ("Custom Code", "LlamaIndex"), key="framework_choice")
    st.selectbox("Choose Language Model:", ("OpenAI (GPT-4o)", "OpenAI (GPT-4o-mini)", "OpenAI (GPT-4.1-mini)", "Llama 3 (via Hugging Face)"), key="llm_choice")
    with st.form("api_key_form"):
        api_key_input = st.text_input(f"Enter {st.session_state.llm_choice} API Key", type="password")
        submitted = st.form_submit_button("Submit & Authenticate Key")
        if submitted:
            with st.spinner("Authenticating..."):
                is_valid = llm_clients.verify_api_key(st.session_state.llm_choice, api_key_input)
                if is_valid:
                    st.session_state.auth_status[st.session_state.llm_choice] = api_key_input
                    st.success("✅ Authenticated!")
                else:
                    st.session_state.auth_status[st.session_state.llm_choice] = None
                    st.error("❌ Authentication Failed.")
    if st.session_state.auth_status.get(st.session_state.llm_choice):
        st.success(f"✅ Key for {st.session_state.llm_choice} is active.")
    else:
        st.warning(f"⚠️ Key for {st.session_state.llm_choice} is not set or invalid.")

# --- KNOWLEDGE BASE LOADING ---
@st.cache_resource(show_spinner="Initializing Custom Knowledge Bases...")
def load_custom_kbs():
    # ... (This function remains the same)
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
st.caption(f"Using: **{st.session_state.framework_choice}** with **{st.session_state.llm_choice}**. For full analysis, include 'show legal precedent'.")

# Load the appropriate KB based on choice
if st.session_state.framework_choice == "Custom Code":
    knowledge_bases = load_custom_kbs()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "thought_process" in msg:
            with st.expander("Show Self-Correction Process"):
                st.info(f"**Sources:** {', '.join(msg['thought_process']['sources'])}")
                st.warning(f"**Initial Draft:**\n{msg['thought_process']['initial']}")
                st.error(f"**Critique:**\n{msg['thought_process']['critique']}")
        if "full_analysis" in msg:
            with st.expander("Show Full Agentic Analysis"):
                st.code(f"Agent's Plan:\n{msg['full_analysis']['plan']}", language="text")
                st.success(f"**Legal & Web Analysis:**\n{msg['full_analysis']['agent_response']}")

if prompt := st.chat_input("Ask about healthcare tax rules..."):
    active_api_key = st.session_state.auth_status.get(st.session_state.llm_choice)
    if not active_api_key:
        st.info("Please submit a valid API key in the sidebar.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        message_to_save = {"role": "assistant"}
        
        # --- CONDITIONAL WORKFLOW LOGIC ---
        if 'show legal precedent' in prompt.lower():
            if st.session_state.framework_choice == "Custom Code":
                with st.spinner("Custom agent is performing full analysis..."):
                    results = custom_agent.run_healthcare_tax_agent(prompt, knowledge_bases, st.session_state.llm_choice, active_api_key)
                    direct_results = results['direct_answer_results']
                    final_response = f"**Direct Answer from IRS Rules:**\n{direct_results['final']}\n\n---\n\n**Further Analysis:**\n{results['cases_answer']}\n{results['web_search_answer']}"
                    message_to_save["thought_process"] = direct_results
                    message_to_save["full_analysis"] = {"plan": results['plan'], "agent_response": f"{results['cases_answer']}\n{results['web_search_answer']}"}
            else: # LlamaIndex
                with st.spinner("LlamaIndex agent is performing full analysis..."):
                    direct_results, agent_response = LlamaIndex_agent.run_llama_index_agent(prompt, st.session_state.llm_choice, active_api_key)
                    final_response = f"**Direct Answer from IRS Rules:**\n{direct_results['final']}\n\n---\n\n**Further Analysis:**\n{agent_response}"
                    message_to_save["thought_process"] = direct_results
                    message_to_save["full_analysis"] = {"plan": "Executed via LlamaIndex ReAct Agent", "agent_response": agent_response}
        else: # Default direct answer
            if st.session_state.framework_choice == "Custom Code":
                with st.spinner("Custom agent is preparing a direct answer..."):
                    results = custom_agent.run_direct_rag_answer(prompt, knowledge_bases, st.session_state.llm_choice, active_api_key)
                    final_response = results['final']
                    message_to_save["thought_process"] = results
            else: # LlamaIndex
                with st.spinner("LlamaIndex is preparing a direct answer..."):
                    # --- THIS LINE IS REMOVED ---
                    # llm = OpenAI(model=st.session_state.llm_choice.split('(')[1].split(')')[0].strip(), api_key=active_api_key)
                    
                    indexes = LlamaIndex_agent.load_llama_index_kbs()
                    # The 'run_direct_llama_index_query' function handles the LLM creation internally
                    results = LlamaIndex_agent.run_direct_llama_index_query(prompt, st.session_state.llm_choice, active_api_key, indexes)
                    final_response = results['final']
                    message_to_save["thought_process"] = results
        
        message_to_save["content"] = final_response
        st.session_state.messages.append(message_to_save)
        st.rerun()