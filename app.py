# app.py
import streamlit as st
import faiss
import pickle
import os
from modules import agentic_core, data_acquisition, retriever, llm_clients

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Healthcare Taxation Assistant", page_icon="⚕️", layout="wide")

# --- Initialize Session State ---
if "auth_status" not in st.session_state:
    st.session_state.auth_status = {}
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "OpenAI (GPT-4o)"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")

    llm_choice = st.selectbox(
        "Choose your Language Model:",
        ("OpenAI (GPT-4o)", "OpenAI (GPT-4o-mini)", "OpenAI (GPT-4.1-mini)", "Llama 3 (via Hugging Face)"),
        key="llm_choice"
    )

    # Use a form for API key submission and authentication
    with st.form("api_key_form"):
        api_key_input = st.text_input(f"Enter {st.session_state.llm_choice} API Key", type="password")
        submitted = st.form_submit_button("Submit & Authenticate Key")

        if submitted:
            with st.spinner("Authenticating key..."):
                # Call the verification function from the llm_clients module
                is_valid = llm_clients.verify_api_key(st.session_state.llm_choice, api_key_input)
                if is_valid:
                    # Store the valid key in a dictionary keyed by the model choice
                    st.session_state.auth_status[st.session_state.llm_choice] = api_key_input
                    st.success("✅ Authenticated!")
                else:
                    st.session_state.auth_status[st.session_state.llm_choice] = None
                    st.error("❌ Authentication Failed. Check your key or service status.")
    
    # Display current authentication status for the selected model
    if st.session_state.auth_status.get(st.session_state.llm_choice):
        st.success(f"✅ Key for {st.session_state.llm_choice} is active.")
    else:
        st.warning(f"⚠️ Key for {st.session_state.llm_choice} is not set or invalid.")

# --- KNOWLEDGE BASE LOADING ---
@st.cache_resource(show_spinner="Initializing Knowledge Bases...")
def load_all_knowledge_bases():
    knowledge_bases = {'irs': (None, None), 'cases': (None, None)}
    if os.path.exists("knowledge_stores/irs_faiss_index.bin"):
        irs_index = faiss.read_index("knowledge_stores/irs_faiss_index.bin")
        with open("knowledge_stores/irs_chunks.pkl", "rb") as f:
            irs_chunks = pickle.load(f)
        knowledge_bases['irs'] = (irs_chunks, irs_index)
    if os.path.exists("knowledge_stores/cases_faiss_index.bin"):
        cases_index = faiss.read_index("knowledge_stores/cases_faiss_index.bin")
        with open("knowledge_stores/cases_chunks.pkl", "rb") as f:
            cases_chunks = pickle.load(f)
        knowledge_bases['cases'] = (cases_chunks, cases_index)
    return knowledge_bases

# --- MAIN APP INTERFACE ---
st.title("⚕️ Healthcare Taxation Assistant")
st.caption(f"An agentic assistant powered by {st.session_state.llm_choice}. For a full analysis, include 'show legal precedent' in your query.")

knowledge_bases = load_all_knowledge_bases()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you with healthcare taxation questions today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about healthcare tax rules..."):
    # Check for authentication status of the currently selected model
    active_api_key = st.session_state.auth_status.get(st.session_state.llm_choice)
    if not active_api_key:
        st.info("Please submit a valid API key in the sidebar to continue.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        if 'show legal precedent' in prompt.lower():
            with st.spinner(f"Performing full agentic analysis with {st.session_state.llm_choice}..."):
                results = agentic_core.run_healthcare_tax_agent(
                    prompt, knowledge_bases, st.session_state.llm_choice, active_api_key
                )
                final_response = f"""
                Here is a full analysis of your query:
                ### 1. Direct Answer from IRS Rules
                {results['direct_answer']}
                *Sources: {', '.join(results['direct_sources']) if results['direct_sources'] else 'N/A'}*
                ---
                ### 2. Relevant Legal Precedents
                {results['cases_answer']}
                *Sources: {', '.join(results['cases_sources']) if results['cases_sources'] else 'N/A'}*
                ---
                ### 3. External Opinions & Analysis
                {results['web_search_answer']}
                """
                st.markdown(final_response)
                with st.expander("Show Agent's Plan"):
                    st.text(results['plan'])
        else:
            with st.spinner(f"Searching IRS documents with {st.session_state.llm_choice}..."):
                results = agentic_core.run_direct_rag_answer(
                    prompt, knowledge_bases, st.session_state.llm_choice, active_api_key
                )
                final_response = f"""
                {results['answer']}
                *Sources: {', '.join(results['sources']) if results['sources'] else 'N/A'}*
                """
                st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})