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

def select_random_pdf(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if pdf_files:
        return os.path.join(pdf_folder, random.choice(pdf_files))
    return None

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

    if is_introduction:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the provided context, generate a one-line introduction about yourself as the patient described in the physiotherapy case study. Use first-person perspective. Include only your name and your primary complaint or condition. Be very concise and disclose minimal information. Do not mention any specific diagnosis.
            <context>
            {context}
            </context>
            """
        )
    elif is_diagnosis:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the provided context, what is the correct diagnosis for this case? Provide only the diagnosis name without any explanation.
            <context>
            {context}
            </context>
            """
        )
    else:
        diagnosis_keywords = [
            "diagnosis", "condition", "what do i have", "what's wrong", "what is wrong",
            "what could it be", "what is it", "what's causing", "what is causing",
            "why do i feel", "reason for", "explanation for", "what's the problem",
            "what is the problem", "what might be wrong", "possible cause",
            "potential issue", "underlying condition", "medical explanation",
            "professional opinion", "expert view", "clinical assessment",
            "what's your take", "what do you think it is", "likely cause",
            "probable condition", "suspected issue", "tentative diagnosis",
            "differential diagnosis", "working diagnosis", "preliminary assessment",
            "initial impression", "diagnostic impression", "clinical impression",
            "provisional diagnosis", "presumptive diagnosis", "diagnostic hypothesis",
            "what's your diagnosis", "can you diagnose", "your professional assessment",
            "clinical opinion", "medical opinion", "diagnostic opinion",
            "what's causing the pain", "reason for the symptoms", "explain my condition"
        ]
        
        if any(keyword in user_input.lower() for keyword in diagnosis_keywords):
            return "I'm not sure about the diagnosis. That's why I'm here to see a physiotherapist. Could you please explain what you think based on what I've told you about my symptoms?", 0

        prompt = ChatPromptTemplate.from_template(
            """
            You are the patient described in the physiotherapy case study. Answer the question from your perspective, using first-person language. 
            Provide a concise response in one or two sentences. If the exact information is not available, 
            use the context to provide a plausible answer based on your condition and experiences.
            Important: Do not mention or reveal any specific diagnosis in your response, even if it's mentioned in the context.
            Do not suggest or speculate about possible diagnoses or underlying conditions.
            If the question seems to be asking for a diagnosis or explanation of your condition in any way, respond with:
            "I'm not sure about the diagnosis. That's why I'm here to see a physiotherapist. Could you please explain what you think based on what I've told you about my symptoms?"
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

    return response['answer'], end - start

# Main app
def main():
    st.title("PhysioPlay")

    pdf_folder = './data/'  # Update this to your PDF folder path

    if not st.session_state.processed_pdf:
        with st.spinner('Selecting and processing a random PDF... This may take a few minutes.'):
            st.session_state.selected_pdf = select_random_pdf(pdf_folder)
            if st.session_state.selected_pdf:
                st.session_state.vectors = process_pdf(st.session_state.selected_pdf)
                st.session_state.processed_pdf = True
                st.success(f"PDF processed successfully: {os.path.basename(st.session_state.selected_pdf)}")
                st.session_state.asked_if_ready = False
            else:
                st.error("No PDF files found in the specified folder.")
                return

    # Ask if ready to start (immediately after processing)
    if st.session_state.processed_pdf and not st.session_state.asked_if_ready:
        st.session_state.chat_history.append({"role": "assistant", "content": "A case study has been randomly selected. Are you ready to start?"})
        st.session_state.asked_if_ready = True

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Your response:")

    if user_input:
        # Display user message
        st.chat_message("user").markdown(user_input)
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if not st.session_state.ready_to_start:
            # Check if user is ready to start
            if any(word in user_input.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'ready']):
                st.session_state.ready_to_start = True
                with st.spinner('Generating case introduction...'):
                    # Generate case introduction
                    introduction, _ = get_chatgroq_response("", is_introduction=True)
                    st.session_state.case_introduction = introduction
                    # Get correct diagnosis (but don't display it)
                    st.session_state.correct_diagnosis, _ = get_chatgroq_response("", is_diagnosis=True)
                
                # Display case introduction
                st.chat_message("assistant").markdown(f"Great! Let's begin. Here's your case:\n\n{st.session_state.case_introduction}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"Great! Let's begin. Here's your case:\n\n{st.session_state.case_introduction}"})
            else:
                st.chat_message("assistant").markdown("Okay, let me know when you're ready to start.")
                st.session_state.chat_history.append({"role": "assistant", "content": "Okay, let me know when you're ready to start."})
        else:
            with st.spinner('Thinking...'):
                response, response_time = get_chatgroq_response(user_input)

            # Display assistant response
            st.chat_message("assistant").markdown(response)
            st.caption(f"Response time: {response_time:.2f} seconds")

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Check if user wants to submit a diagnosis
            if "diagnosis" in user_input.lower() and not st.session_state.diagnosis_revealed:
                st.session_state.diagnosis_submitted = True

    # Handle diagnosis submission
    if st.session_state.diagnosis_submitted and not st.session_state.diagnosis_revealed:
        user_diagnosis = st.text_input("What do you think the diagnosis is?")
        if user_diagnosis:
            if user_diagnosis.lower() == st.session_state.correct_diagnosis.lower():
                st.success("Correct diagnosis!")
            else:
                st.error(f"Incorrect. The correct diagnosis is: {st.session_state.correct_diagnosis}")
            st.session_state.diagnosis_revealed = True
            st.info(f"Case study used: {os.path.basename(st.session_state.selected_pdf)}")

if __name__ == "__main__":
    main()