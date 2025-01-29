import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import tempfile
import time
import os
import random
import re

# Set page config
st.set_page_config(page_title="PhysioPlay", layout="wide")

# Enhanced CSS for fixed button positioning
st.markdown("""
    <style>
    /* Container for the diagnosis button */
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
    
    /* Style for the diagnosis button */
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    
    /* Hover effect for the button */
    .stButton button:hover {
        background-color: #ff3333;
    }
    
    /* Ensure chat input stays at bottom */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px;
        z-index: 998;
    }
    
    /* Add some padding at the bottom to prevent chat history from being hidden */
    .main {
        padding-bottom: 150px;
    }
    </style>
""", unsafe_allow_html=True)

# Hardcoded API key
GROQ_API_KEY = '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'

# Enhanced list of diagnostic/medical keywords and patterns
DIAGNOSTIC_PATTERNS = {
    'direct_diagnosis_questions': [
        r'what.*(?:wrong|problem|condition|diagnosis|issue)',
        r'(?:tell|explain).*(?:wrong|problem|condition|diagnosis)',
        r'(?:what|why).*(?:cause|reason)',
        r'(?:what|do).*(?:think|believe)',
        r'could.*(?:be|have)',
        r'is.*(?:it|this)',
    ],
    'medical_terms': [
        r'\b(?:arthritis|tendinitis|bursitis|sprain|strain|tear|rupture|fracture|dislocation)\b',
        r'\b(?:syndrome|disorder|disease|condition|pathology|dysfunction|impingement)\b',
        r'\b(?:lateral|medial|anterior|posterior|proximal|distal|bilateral)\b',
    ],
    'body_parts': [
        r'\b(?:shoulder|elbow|wrist|hip|knee|ankle|spine|back|neck|joint)\b',
        r'\b(?:muscle|tendon|ligament|bone|cartilage|nerve|tissue)\b',
    ],
    'assessment_terms': [
        r'\b(?:assess|evaluate|diagnose|examine|test|check)\b',
        r'(?:what|how).*(?:assessment|evaluation|diagnosis|examination)',
    ]
}

# Initialize session state
if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = False
    st.session_state.vectors = None
    st.session_state.chat_history = []
    st.session_state.case_introduction = ""
    st.session_state.asked_if_ready = False
    st.session_state.ready_to_start = False
    st.session_state.diagnosis_revealed = False
    st.session_state.correct_diagnosis = ""
    st.session_state.selected_pdf = None
    st.session_state.diagnosis_submitted = False
    st.session_state.pdf_name = None
    st.session_state.show_diagnosis_input = False

def normalize_text(text):
    """Normalize text for comparison."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = ' '.join(text.split())
    return text

def extract_primary_diagnosis(diagnosis_text):
    """Extract only the primary diagnosis without any additional information."""
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
    
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text.strip()

def select_random_pdf(pdf_folder):
    """Select a random PDF file from the specified folder."""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if pdf_files:
        selected_file = random.choice(pdf_files)
        return os.path.join(pdf_folder, selected_file), selected_file
    return None, None

def process_pdf(pdf_path):
    """Process the PDF file and create a vector store."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def is_diagnostic_question(text):
    """
    Enhanced function to detect if a question is attempting to elicit diagnostic information.
    Returns (bool, str) tuple - is_diagnostic and the reason why
    """
    text = text.lower()
    
    # Check against each pattern category
    for category, patterns in DIAGNOSTIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return True, f"Detected {category} pattern"
    
    return False, ""

def get_patient_deflection():
    """
    Returns a randomized patient deflection response when users ask about diagnosis
    """
    deflections = [
        "I'm not sure about all that medical stuff. That's why I'm here to see you.",
        "I don't really understand the medical terms. Can you explain what you mean?",
        "I just know how it feels - I'll leave the medical details to you.",
        "That's a bit technical for me. I just want to feel better.",
        "I wouldn't know about that - I'm hoping you can tell me what's wrong.",
        "I'm not familiar with medical terminology. Could you ask me about my symptoms instead?",
    ]
    return random.choice(deflections)

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    """Generate a response using the ChatGroq model with enhanced question filtering."""
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
            
            <context>
            {context}
            </context>
            """
        )
    elif is_diagnosis:
        prompt = ChatPromptTemplate.from_template(
            """
            Extract ONLY the primary diagnosis from the case study. 
            Provide ONLY the basic medical condition name without any qualifiers, descriptions, or additional details.
            Do NOT include secondary diagnoses, descriptors like 'bilateral', 'chronic', etc., or any other information.
            
            <context>
            {context}
            </context>
            """
        )
    else:
        # Check if the question is trying to elicit diagnostic information
        is_diagnostic, _ = is_diagnostic_question(user_input)
        if is_diagnostic:
            return get_patient_deflection(), 0

        prompt = ChatPromptTemplate.from_template(
            """
            Act as the patient. Follow these STRICT rules:
            1. NEVER suggest or hint at any diagnosis
            2. NEVER use medical terminology
            3. NEVER mention specific body parts or conditions
            4. ONLY describe current feelings and symptoms in simple terms
            5. Keep responses to ONE short sentence
            6. Use everyday language a patient would use
            7. If asked about medical terms or diagnoses, express confusion or defer to the physiotherapist
            
            <context>
            {context}
            </context>
            Question: {input}
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
    """Main application function with improved UI and persistent diagnosis button."""
    st.title("PhysioPlay")

    pdf_folder = 'C:/Users/Vivek Vasani/Desktop/PhysioPlay/data'

    # Initialize PDF processing
    if not st.session_state.processed_pdf:
        with st.spinner('Selecting and processing a random case study... This may take a few minutes.'):
            selected_pdf_path, pdf_name = select_random_pdf(pdf_folder)
            if selected_pdf_path:
                st.session_state.selected_pdf = selected_pdf_path
                st.session_state.pdf_name = pdf_name
                st.session_state.vectors = process_pdf(selected_pdf_path)
                st.session_state.processed_pdf = True
                st.success("Case study loaded successfully!")
                st.session_state.asked_if_ready = False
            else:
                st.error("No case studies found in the specified folder.")
                return

    # Create main containers
    chat_container = st.container()
    
    # Create a placeholder for the diagnosis button
    button_placeholder = st.empty()
    
    # Display chat history
    with chat_container:
        if st.session_state.processed_pdf and not st.session_state.asked_if_ready:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "A case study has been randomly selected. Are you ready to start?"
            })
            st.session_state.asked_if_ready = True

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Display the persistent diagnosis button
    if st.session_state.ready_to_start and not st.session_state.diagnosis_revealed:
        with button_placeholder:
            st.markdown(
                """
                <div class="diagnosis-button-container">
                    <div class="stButton">
                        <button kind="primary">Ready with diagnosis</button>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Ready with diagnosis", key="diagnosis_button"):
                st.session_state.show_diagnosis_input = True

    # Show diagnosis input if button was clicked
    if st.session_state.show_diagnosis_input and not st.session_state.diagnosis_revealed:
        diagnosis_container = st.container()
        with diagnosis_container:
            user_diagnosis = st.text_input("What do you think the diagnosis is?")
            if user_diagnosis:
                normalized_user_diagnosis = normalize_text(user_diagnosis)
                normalized_correct_diagnosis = normalize_text(st.session_state.correct_diagnosis)
                
                if normalized_user_diagnosis == normalized_correct_diagnosis:
                    st.success("Correct diagnosis!")
                    st.info(f"Case Study: {st.session_state.pdf_name}")
                else:
                    st.error(f"Incorrect. The correct diagnosis is: {st.session_state.correct_diagnosis}")
                    st.info(f"Case Study: {st.session_state.pdf_name}")
                st.session_state.diagnosis_revealed = True

    # Handle user input
    user_input = st.chat_input("Your response:")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input
        })

        if not st.session_state.ready_to_start:
            if any(word in user_input.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'ready']):
                st.session_state.ready_to_start = True
                with st.spinner('Preparing your case...'):
                    introduction, _ = get_chatgroq_response("", is_introduction=True)
                    st.session_state.case_introduction = introduction
                    st.session_state.correct_diagnosis, _ = get_chatgroq_response("", is_diagnosis=True)
                
                response_text = f"Great! Let's begin.\n\n{st.session_state.case_introduction}"
                st.chat_message("assistant").markdown(response_text)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response_text
                })
            else:
                response_text = "Okay, let me know when you're ready to start."
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