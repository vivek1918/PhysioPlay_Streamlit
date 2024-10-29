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
st.set_page_config(page_title="Physiotherapy Case Study Practice", layout="wide")

# Hardcoded API key
GROQ_API_KEY = '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'

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

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    """Generate a response using the ChatGroq model."""
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
        # Expanded list of keywords that trigger the default response
        restricted_keywords = [
            "diagnosis", "condition", "problem", "issue", "wrong", "cause",
            "hip", "knee", "back", "spine", "joint", "muscle", "bone",
            "arthritis", "strain", "sprain", "tear", "inflammation",
            "why", "how come", "what's", "whats", "tell", "explain",
            "think", "believe", "suspect", "possible", "maybe", "might",
            "could", "would", "assess", "evaluate"
        ]
        
        if any(keyword in user_input.lower() for keyword in restricted_keywords):
            return "I can't tell you that. That's why I am here to see a physiotherapist.", 0

        prompt = ChatPromptTemplate.from_template(
            """
            Act as the patient. Follow these STRICT rules:
            1. NEVER suggest or hint at any diagnosis
            2. NEVER use medical terminology
            3. NEVER mention specific body parts or conditions
            4. ONLY describe current feelings and symptoms in simple terms
            5. Keep responses to ONE short sentence
            
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
    """Main application function."""
    st.title("Physiotherapy Case Study Practice")

    pdf_folder = './data/'

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

    # Display chat history and handle interactions
    if st.session_state.processed_pdf and not st.session_state.asked_if_ready:
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": "A case study has been randomly selected. Are you ready to start?"
        })
        st.session_state.asked_if_ready = True

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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

            if "diagnosis" in user_input.lower() and not st.session_state.diagnosis_revealed:
                st.session_state.diagnosis_submitted = True

    if st.session_state.diagnosis_submitted and not st.session_state.diagnosis_revealed:
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

if __name__ == "__main__":
    main()