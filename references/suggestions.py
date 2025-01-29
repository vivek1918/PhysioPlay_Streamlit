import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain, LLMChain
from langchain_community.vectorstores import FAISS
import tempfile
import time
import os
import random
import re
import json

# Set page config
st.set_page_config(page_title="PhysioPlay", layout="wide")

# CSS remains the same as your original code
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

GROQ_API_KEY = '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'

# Only maintain diagnostic patterns for actual diagnosis questions
DIAGNOSTIC_PATTERNS = [
    r'what.*(?:diagnosis|condition|disorder|disease)',
    r'(?:tell|explain).*(?:diagnosis|condition|disorder|disease)',
    r'(?:could|might|may).*(?:diagnosis|condition|disorder|disease)',
]

# Initialize session state variables
if "processed_json" not in st.session_state:
    st.session_state.processed_json = False
    st.session_state.vectors = None
    st.session_state.chat_history = []
    st.session_state.case_introduction = ""
    st.session_state.asked_if_ready = False
    st.session_state.ready_to_start = False
    st.session_state.diagnosis_revealed = False
    st.session_state.correct_diagnosis = ""
    st.session_state.selected_json = None
    st.session_state.diagnosis_submitted = False
    st.session_state.json_name = None
    st.session_state.show_diagnosis_input = False

def normalize_text(text):
    """Normalize text for comparison."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return ' '.join(text.split())

def extract_primary_diagnosis(diagnosis_text):
    """Extract only the primary diagnosis."""
    patterns_to_remove = [
        r'the primary diagnosis is',
        r'likely',
        r'probable',
        r'suspected',
        r'with.*',
        r'secondary.*',
        r'differential.*',
        r'and.*',
        r'possibly.*',
        r'may.*',
        r'could.*',
        r'associated.*'
    ]
    
    cleaned_text = diagnosis_text.lower()
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    return ' '.join(cleaned_text.split()).strip()

def select_random_json(json_folder):
    """Select a random JSON file."""
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    if json_files:
        selected_file = random.choice(json_files)
        return os.path.join(json_folder, selected_file), selected_file
    return None, None

def process_json(json_path):
    """Process JSON and create vector store."""
    with open(json_path, 'r') as file:
        case_data = json.load(file)
    
    # Convert JSON data to a string format for processing
    case_text = json.dumps(case_data)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(case_text)
    
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(splits, embeddings)

def is_diagnosis_question(text):
    """Check if the question is directly asking for a diagnosis."""
    text = text.lower()
    return any(re.search(pattern, text) for pattern in DIAGNOSTIC_PATTERNS)

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    """Generate direct responses using ChatGroq."""
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

    if is_introduction:
        prompt = ChatPromptTemplate.from_template(
            """
            Generate a one-line patient introduction. Include ONLY:
            1. A greeting
            2. First name only
            3. The main symptom in simple terms, without any medical terminology
            
            Format: "Hi, I'm [First Name]. I have [simple description of main symptom]."
            
            Do NOT include:
            - Any medical terms
            - Location of pain/symptoms
            - Duration of symptoms
            - Any other details
            
            Context: {context}
            """
        )
    elif is_diagnosis:
        prompt = ChatPromptTemplate.from_template(
            """
            Extract ONLY the primary diagnosis from the case study. 
            Provide ONLY the basic medical condition name without any qualifiers, descriptions, or additional details.
            Do NOT include secondary diagnoses, descriptors like 'bilateral', 'chronic', etc., or any other information.
            
            Context: {context}
            """
        )
    else:
        # Only deflect direct diagnosis questions
        if is_diagnosis_question(user_input):
            return "I'm not sure about the exact medical condition - that's why I'm here to see you.", 0

        prompt = ChatPromptTemplate.from_template(
            """
            You are a patient speaking to a physiotherapist. Based on the case study context,
            respond directly to the question while following these rules:

            1. Only provide information about symptoms, medical history, lifestyle, and other background details as per the case study data.
            2. Do not offer any potential diagnoses, medical terms, or suggestions that could help the physiotherapist guess the diagnosis.
            3. If asked about your opinion on the cause of the symptoms, respond by saying that you are not a medical professional and cannot speculate on the diagnosis.
            4. If the physiotherapist directly asks for a diagnosis or specific medical terms, politely decline and reiterate your role as a patient.
            5. Your responses should mimic a real patient's natural language and uncertainty about medical matters.
            6. Always respond within the context of the case study without revealing information that is not explicitly asked about in symptoms or history.
            7. Keep responses to ONE short sentence.

            Remember, you are not aware of the diagnosis or medical details beyond the symptoms and history described in the case study.
            Also remember, you will speak as the patient in first person and talk to the physiotherapist in terms of a conversation in the present tense.
            Context: {context}
            Question: {input}
            
            Respond naturally and directly:
            """
        )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    end = time.process_time()

    if is_diagnosis:
        response['answer'] = extract_primary_diagnosis(response['answer'])

    return response['answer'], end - start

def main():
    """Main application function."""
    st.title("PhysioPlay")

    json_folder = 'json_data'

    # Initialize JSON processing
    if not st.session_state.processed_json:
        with st.spinner('Loading a new patient case...'):
            selected_json_path, json_name = select_random_json(json_folder)
            if selected_json_path:
                st.session_state.selected_json = selected_json_path
                st.session_state.json_name = json_name
                st.session_state.vectors = process_json(selected_json_path)
                st.session_state.processed_json = True
                st.success("New patient case ready!")
                st.session_state.asked_if_ready = False
            else:
                st.error("No case studies found in the specified folder.")
                return

    # Create containers
    chat_container = st.container()
    button_placeholder = st.empty()
    
    # Display chat history
    with chat_container:
        if st.session_state.processed_json and not st.session_state.asked_if_ready:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "Your next patient has arrived. Would you like to begin the consultation?"
            })
            st.session_state.asked_if_ready = True

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Display diagnosis button
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

    # Handle diagnosis submission
    if st.session_state.show_diagnosis_input and not st.session_state.diagnosis_revealed:
        diagnosis_container = st.container()
        with diagnosis_container:
            user_diagnosis = st.text_input("What is your diagnosis?")
            if user_diagnosis:
                if normalize_text(user_diagnosis) == normalize_text(st.session_state.correct_diagnosis):
                    st.success("Correct diagnosis!")
                else:
                    st.error(f"Incorrect. The correct diagnosis was: {st.session_state.correct_diagnosis}")
                st.info(f"Case Study: {st.session_state.json_name}")
                st.session_state.diagnosis_revealed = True

    # Handle user input
    user_input = st.chat_input("Ask your patient a question...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input
        })

        if not st.session_state.ready_to_start:
            if any(word in user_input.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'ready']):
                st.session_state.ready_to_start = True
                with st.spinner('Getting patient information...'):
                    introduction, _ = get_chatgroq_response("", is_introduction=True)
                    st.session_state.case_introduction = introduction
                    st.session_state.correct_diagnosis, _ = get_chatgroq_response("", is_diagnosis=True)
                
                st.chat_message("assistant").markdown(st.session_state.case_introduction)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": st.session_state.case_introduction
                })
            else:
                response = "Let me know when you're ready to see the patient."
                st.chat_message("assistant").markdown(response)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response
                })
        else:
            # Generate the patient's response
            with st.spinner('Thinking...'):
                response, response_time = get_chatgroq_response(user_input)

            st.chat_message("assistant").markdown(response)
            st.caption(f"Response time: {response_time:.2f} seconds")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response
            })

            # Suggest a follow-up question
            context_prompt = PromptTemplate(
                template="Based on the patient's last response: '{last_response}', suggest a follow-up question a physiotherapist might ask that a patient will understand. Make note the suggested question should be a ONE line question only.",
                input_variables=["last_response"]
            )
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
            context_llm_chain = LLMChain(llm=llm, prompt=context_prompt)
            follow_up_question = context_llm_chain.run({"last_response": response})
            st.write("Suggested Follow-Up Question:", follow_up_question)

if __name__ == "__main__":
    main()