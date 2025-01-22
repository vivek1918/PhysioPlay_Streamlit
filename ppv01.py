import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import json
import time
import os
import random
import re

# Set page config
st.set_page_config(page_title="PhysioPlay", layout="wide")

# CSS for UI elements
st.markdown("""
    <style>
    .diagnosis-button-container {
        position: fixed;
        bottom: 80px;
        right: 20px;
        z-index: 999;
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: #ff3333;
    }
    
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px;
        z-index: 998;
    }
    
    .main {
        padding-bottom: 150px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize API key
GROQ_API_KEY = st.secrets['GROQ_API_KEY']

# Keywords to detect diagnostic questions
DIAGNOSTIC_KEYWORDS = [
    r'what.*(?:wrong|problem|condition|diagnosis)',
    r'(?:tell|explain).*(?:problem|condition)',
    r'(?:what|why).*(?:cause|reason)',
    r'could.*(?:be|have)',
    r'is.*(?:it|this)',
]

# Initialize session state
if "case_loaded" not in st.session_state:
    st.session_state.case_loaded = False
    st.session_state.vectors = None
    st.session_state.chat_history = []
    st.session_state.case_introduction = ""
    st.session_state.asked_if_ready = False
    st.session_state.ready_to_start = False
    st.session_state.diagnosis_revealed = False
    st.session_state.correct_diagnosis = ""
    st.session_state.selected_case = None
    st.session_state.case_name = None
    st.session_state.show_diagnosis_input = False

def normalize_text(text):
    """Normalize text for comparison."""
    return ' '.join(re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split())

def select_random_case(json_folder):
    """Select a random case file from the JSON folder."""
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    if json_files:
        selected_file = random.choice(json_files)
        return os.path.join(json_folder, selected_file), selected_file
    return None, None

def load_case_data(json_path):
    """Load and process the JSON case file."""
    with open(json_path, 'r') as file:
        case_data = json.load(file)
    
    # Convert case data to format suitable for vector store
    documents = [{"page_content": str(case_data)}]
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts([doc["page_content"] for doc in documents], embeddings)
    
    return vectorstore

def is_diagnostic_question(text):
    """Check if the question is attempting to get diagnostic information."""
    text = text.lower()
    return any(re.search(pattern, text) for pattern in DIAGNOSTIC_KEYWORDS)

def get_patient_response():
    """Get a deflection response when users ask about diagnosis."""
    responses = [
        "I just know it hurts - I'm hoping you can help me understand what's wrong.",
        "That sounds too technical for me. Can you ask me about how I feel instead?",
        "I don't really know about medical stuff. That's why I'm here.",
        "Could you ask me about my symptoms instead?",
    ]
    return random.choice(responses)

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    """Generate response using the ChatGroq model."""
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

    if is_introduction:
        prompt = ChatPromptTemplate.from_template("""
            Provide a very brief patient introduction in exactly 2 lines:
            Line 1: "Hi, I'm [First Name]."
            Line 2: State only the primary symptom in simple terms.
            Don't mention duration, medical terms, or any other details.
            
            Context: {context}
        """)
    elif is_diagnosis:
        prompt = ChatPromptTemplate.from_template("""
            Extract only the primary diagnosis from the case.
            Provide just the basic condition name without any qualifiers.
            
            Context: {context}
        """)
    else:
        if is_diagnostic_question(user_input):
            return get_patient_response(), 0

        prompt = ChatPromptTemplate.from_template("""
            Respond as the patient described in the case. Rules:
            1. Always remember this is like a game, the user is trying to diagnose as per the persona by asking questions, so never ever spill out the diagnosis.
            2. Use only simple language
            3. Describe only how you feel or what you experience
            4. Keep responses brief and natural
            5. If asked about medical terms, help with clues but never spill out the diagnosis.
            6. Reveal results of x-ray, mri or special tests or any other diagnostic tests if asked just make sure to not spill out the diagnosis.
            
            Context: {context}
            Question: {input}
        """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    end = time.process_time()

    return response['answer'], end - start

def main():
    """Main application function."""
    st.title("PhysioPlay")

    json_folder = './json_data/'

    # Initialize case loading
    if not st.session_state.case_loaded:
        with st.spinner('Loading a random case...'):
            selected_case_path, case_name = select_random_case(json_folder)
            if selected_case_path:
                st.session_state.selected_case = selected_case_path
                st.session_state.case_name = case_name
                st.session_state.vectors = load_case_data(selected_case_path)
                st.session_state.case_loaded = True
                st.success("Case loaded successfully!")
                st.session_state.asked_if_ready = False
            else:
                st.error("No cases found in the specified folder.")
                return

    chat_container = st.container()
    button_placeholder = st.empty()
    
    with chat_container:
        if st.session_state.case_loaded and not st.session_state.asked_if_ready:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "A case has been selected. Ready to begin?"
            })
            st.session_state.asked_if_ready = True

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.ready_to_start and not st.session_state.diagnosis_revealed:
        with button_placeholder:
            st.markdown(
                """
                <div class="diagnosis-button-container">
                    <div class="stButton">
                        <button kind="primary">Submit Diagnosis</button>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Submit Diagnosis", key="diagnosis_button"):
                st.session_state.show_diagnosis_input = True

    if st.session_state.show_diagnosis_input and not st.session_state.diagnosis_revealed:
        diagnosis_container = st.container()
        with diagnosis_container:
            user_diagnosis = st.text_input("Enter your diagnosis:")
            if user_diagnosis:
                if normalize_text(user_diagnosis) == normalize_text(st.session_state.correct_diagnosis):
                    st.success("Correct diagnosis!")
                else:
                    st.error(f"Incorrect. The correct diagnosis was: {st.session_state.correct_diagnosis}")
                st.info(f"Case: {st.session_state.case_name}")
                st.session_state.diagnosis_revealed = True

    user_input = st.chat_input("Your message:")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input
        })

        if not st.session_state.ready_to_start:
            if any(word in user_input.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'ready']):
                st.session_state.ready_to_start = True
                with st.spinner('Preparing case...'):
                    introduction, _ = get_chatgroq_response("", is_introduction=True)
                    st.session_state.case_introduction = introduction
                    st.session_state.correct_diagnosis, _ = get_chatgroq_response("", is_diagnosis=True)
                
                response_text = f"Let's begin!\n\n{st.session_state.case_introduction}"
                st.chat_message("assistant").markdown(response_text)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response_text
                })
            else:
                response_text = "Let me know when you're ready to start."
                st.chat_message("assistant").markdown(response_text)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response_text
                })
        else:
            with st.spinner('Thinking...'):
                response, response_time = get_chatgroq_response(user_input)

            st.chat_message("assistant").markdown(response)
            st.caption(f"Response time: {response_time:.2f} seconds")

            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response
            })

if __name__ == "__main__":
    main()