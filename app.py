import streamlit as st
import faiss
import pickle
import os
from modules import agentic_core, data_acquisition, retriever, llm_clients

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Healthcare Taxation Assistant",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
# This block must be at the top to prevent AttributeErrors
if "auth_status" not in st.session_state:
    st.session_state.auth_status = {}
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "OpenAI (GPT-4o)" # Default model
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you with healthcare taxation questions today?"}]


# --- SIDEBAR FOR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model Selection Dropdown
    st.selectbox(
        "Choose your Language Model:",
        ("OpenAI (GPT-4o)", "OpenAI (GPT-4o-mini)", "OpenAI (GPT-4.1-mini)", "Llama 3 (via Hugging Face)"),
        key="llm_choice" # Binds the selection to the session state
    )

    # Authentication Form
    with st.form("api_key_form", clear_on_submit=True):
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

    # Display current authentication status
    if st.session_state.auth_status.get(st.session_state.llm_choice):
        st.success(f"✅ Key for {st.session_state.llm_choice} is active.")
    else:
        st.warning(f"⚠️ Key for {st.session_state.llm_choice} is not set or invalid.")


# --- KNOWLEDGE BASE LOADING ---
@st.cache_resource(show_spinner="Initializing Knowledge Bases...")
def load_or_build_knowledge_base():
    index_path_irs = "knowledge_stores/irs_faiss_index.bin"
    chunks_path_irs = "knowledge_stores/irs_chunks.pkl"
    index_path_cases = "knowledge_stores/cases_faiss_index.bin"
    chunks_path_cases = "knowledge_stores/cases_chunks.pkl"

    knowledge_bases = {'irs': (None, None), 'cases': (None, None)}

    # Try to load pre-computed files first
    if os.path.exists(index_path_irs) and os.path.exists(chunks_path_irs) and os.path.exists(index_path_cases) and os.path.exists(chunks_path_cases):
        st.sidebar.info("Loading pre-computed knowledge bases from disk.")
        try:
            irs_index = faiss.read_index(index_path_irs)
            with open(chunks_path_irs, "rb") as f: irs_chunks = pickle.load(f)
            knowledge_bases['irs'] = (irs_chunks, irs_index)

            cases_index = faiss.read_index(index_path_cases)
            with open(chunks_path_cases, "rb") as f: cases_chunks = pickle.load(f)
            knowledge_bases['cases'] = (cases_chunks, cases_index)
            st.sidebar.success("All knowledge bases loaded successfully.")
            return knowledge_bases
        except Exception as e:
            st.sidebar.error(f"Error loading files: {e}. Rebuilding...")

    # Fallback: Build from scratch if files are missing
    st.sidebar.warning("One or more knowledge base files not found. Building from scratch...")
    os.makedirs("knowledge_stores", exist_ok=True) # Ensure directory exists
    
    # Build IRS KB
    publication_urls = data_acquisition.load_urls_from_file()
    irs_content = {name: content for name, url in publication_urls.items() if (content := data_acquisition.scrape_publication(name, url))}
    if irs_content:
        irs_chunks, irs_index = retriever.build_faiss_index(irs_content)
        faiss.write_index(irs_index, index_path_irs)
        with open(chunks_path_irs, "wb") as f: pickle.dump(irs_chunks, f)
        knowledge_bases['irs'] = (irs_chunks, irs_index)

    # Build Cases KB
    legal_cases_content = data_acquisition.extract_text_from_pdfs("source_documents/legal_cases")
    if legal_cases_content:
        case_chunks, case_index = retriever.build_faiss_index(legal_cases_content)
        faiss.write_index(case_index, index_path_cases)
        with open(chunks_path_cases, "wb") as f: pickle.dump(case_chunks, f)
        knowledge_bases['cases'] = (case_chunks, case_index)

    st.sidebar.success("Knowledge bases built and saved.")
    return knowledge_bases


# --- MAIN APP INTERFACE ---
st.title("⚕️ Healthcare Taxation Assistant")
st.caption(f"An agentic assistant powered by {st.session_state.llm_choice}. For a full analysis, include 'show legal precedent' in your query.")

knowledge_bases = load_or_build_knowledge_base()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Display thought process if it exists for a message
        if "thought_process" in msg:
            with st.expander("Show Agent's Thought Process"):
                st.subheader("Retrieved Sources")
                st.info(f"**Sources:** {', '.join(msg['thought_process']['sources'])}")
                st.subheader("Initial Answer (Draft)")
                st.warning(msg['thought_process']['initial'])
                st.subheader("AI's Self-Critique")
                st.error(msg['thought_process']['critique'])
        # Display full analysis if it exists
        if "full_analysis" in msg:
            with st.expander("Show Full Agentic Analysis"):
                st.subheader("Agent's Plan")
                st.text(msg['full_analysis']['plan'])
                st.subheader("Legal Precedent Analysis")
                st.markdown(f"{msg['full_analysis']['cases_answer']}\n\n*Sources: {', '.join(msg['full_analysis']['cases_sources']) if msg['full_analysis']['cases_sources'] else 'N/A'}*")
                st.subheader("External Opinions & Links")
                st.markdown(msg['full_analysis']['web_search_answer'])

# Accept and process user input
if prompt := st.chat_input("Ask about healthcare tax rules..."):
    active_api_key = st.session_state.auth_status.get(st.session_state.llm_choice)
    if not active_api_key:
        st.info("Please submit a valid API key in the sidebar to continue.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        message_to_save = {"role": "assistant"}

        if 'show legal precedent' in prompt.lower():
            with st.spinner(f"Performing full agentic analysis with {st.session_state.llm_choice}..."):
                results = agentic_core.run_healthcare_tax_agent(prompt, knowledge_bases, st.session_state.llm_choice, active_api_key)
                direct_answer_results = results['direct_answer']
                
                final_response_text = f"**Direct Answer from IRS Rules:**\n{direct_answer_results['final']}"
                st.markdown(final_response_text)

                message_to_save["content"] = final_response_text
                message_to_save["thought_process"] = {
                    "sources": direct_answer_results['sources'],
                    "initial": direct_answer_results['initial'],
                    "critique": direct_answer_results['critique']
                }
                message_to_save["full_analysis"] = {
                    "plan": results['plan'],
                    "cases_answer": results['cases_answer'],
                    "cases_sources": results['cases_sources'],
                    "web_search_answer": results['web_search_answer']
                }
        else:
            with st.spinner(f"Searching IRS documents and self-correcting with {st.session_state.llm_choice}..."):
                results = agentic_core.run_direct_rag_answer(prompt, knowledge_bases, st.session_state.llm_choice, active_api_key)
                final_response_text = results['final']
                st.markdown(final_response_text)

                message_to_save["content"] = final_response_text
                message_to_save["thought_process"] = {
                    "sources": results['sources'],
                    "initial": results['initial'],
                    "critique": results['critique']
                }
        
        # This will append the assistant's message with all its context to the history
        st.session_state.messages.append(message_to_save)
        # We need to rerun to make the new expanders appear correctly in the history
        st.rerun()