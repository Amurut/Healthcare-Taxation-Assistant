# app.py
import streamlit as st
import faiss
import pickle
import os
from modules import agentic_core, data_acquisition, retriever

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Healthcare Taxation Assistant", page_icon="⚕️", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # MODIFIED: Expanded the list of LLM options
    llm_choice = st.selectbox(
        "Choose your Language Model:",
        (
            "Llama 3 (via Hugging Face)",
            "OpenAI (GPT-4o)",
            "OpenAI (GPT-4o-mini)",
            "OpenAI (GPT-4.1-mini)"
        ),
        key="llm_choice"
    )

    api_key_to_use = None
    if "OpenAI" in st.session_state.llm_choice:
        st.subheader("OpenAI API Key")
        api_key_to_use = st.text_input("Enter OpenAI API Key", key="openai_key", type="password")
    elif "Llama 3" in st.session_state.llm_choice:
        st.subheader("Hugging Face API Key")
        api_key_to_use = st.text_input("Enter HuggingFace Token", key="hf_token", type="password")

# --- KNOWLEDGE BASE LOADING ---
@st.cache_resource(show_spinner="Initializing Knowledge Bases...")
def load_all_knowledge_bases():
    knowledge_bases = {'irs': (None, None), 'cases': (None, None)}
    # Load IRS data
    if os.path.exists("knowledge_stores/irs_faiss_index.bin"):
        irs_index = faiss.read_index("knowledge_stores/irs_faiss_index.bin")
        with open("knowledge_stores/irs_chunks.pkl", "rb") as f:
            irs_chunks = pickle.load(f)
        knowledge_bases['irs'] = (irs_chunks, irs_index)
    # Load Legal Cases data
    if os.path.exists("knowledge_stores/cases_faiss_index.bin"):
        cases_index = faiss.read_index("knowledge_stores/cases_faiss_index.bin")
        with open("knowledge_stores/cases_chunks.pkl", "rb") as f:
            cases_chunks = pickle.load(f)
        knowledge_bases['cases'] = (cases_chunks, cases_index)
    return knowledge_bases

# --- MAIN APP INTERFACE ---
st.title("⚕️ Healthcare Taxation Assistant")
st.caption(f"An agentic assistant powered by {st.session_state.llm_choice}")

knowledge_bases = load_all_knowledge_bases()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you with healthcare taxation questions today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about healthcare tax rules, precedents, etc..."):
    if not api_key_to_use:
        st.info("Please enter your API key in the sidebar.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Agent is planning and researching with {st.session_state.llm_choice}..."):
            # The agentic_core now receives the specific LLM choice
            results = agentic_core.run_healthcare_tax_agent(
                prompt, knowledge_bases, st.session_state.llm_choice, api_key_to_use
            )
            
            final_response = f"""
            Here is a summary based on my research:

            ### 1. IRS Rules & Publications
            {results['irs_answer']}
            *Sources: {', '.join(results['irs_sources']) if results['irs_sources'] else 'N/A'}*

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

    st.session_state.messages.append({"role": "assistant", "content": final_response})