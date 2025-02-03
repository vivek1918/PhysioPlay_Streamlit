import streamlit as st
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain, LLMChain
from langchain_community.vectorstores import FAISS
import json
import time
import os
import random
import re
from textblob import TextBlob
import pyttsx3

# Set page config
st.set_page_config(page_title="PhysioPlay", layout="wide")

# Update the CSS part with these modified styles
st.markdown("""
    <style>
    /* Other styles remain the same... */

    /* Message container styling */
    .stChatMessage {
        display: flex !important;
        align-items: center !important;
        padding-right: 0 !important;
    }
    
    .stChatMessageContent {
        margin-right: 0 !important;
    }
    
    /* Speaker button container */
    .speaker-container {
        display: inline-flex !important;
        align-items: center !important;
        padding-left: 5px !important;
        padding-right: 5px !important;
    }
    
    /* Speaker button styling */
    .speaker-button {
        padding: 0px 8px !important;
        height: 30px !important;
        margin-left: 5px !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    .speaker-button:hover {
        background-color: #f0f0f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Update the display_message function
def display_message(message, index):
    """Display a chat message with an inline speaker button."""
    message_container = st.container()
    with message_container:
        cols = st.columns([0.97, 0.03])  # Adjust these values to fine-tune positioning
        
        with cols[0]:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            else:
                st.chat_message("assistant").markdown(message["content"])
        
        with cols[1]:
            # Place the button in a container that aligns with the message
            btn_container = st.container()
            with btn_container:
                if st.button("🔊", key=f"speaker_{index}", 
                           help="Click to hear message",
                           use_container_width=True):
                    play_text_tts(message["content"], st.session_state.persona_gender)

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
    st.session_state.persona_gender = "female"  # Default persona gender

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

def analyze_sentiment(text):
    """Analyze sentiment and return a simple label."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "Good"
    elif polarity < -0.2:
        return "Bad"
    else:
        return "Neutral"

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    """Generate response using the ChatGroq model."""
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

    if is_introduction:
        prompt = ChatPromptTemplate.from_template("""
            Provide a very brief introduction of yourself as the patient in exactly 2 lines:
            Context: {context}
            Speak in the first person.
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
            1. Speak in the first person, as though you are the patient.
            2. Always remember this is like a game; the user is trying to diagnose you by asking questions, so never ever spill out the diagnosis.
            3. Use only simple language and keep your responses natural and conversational.
            4. Describe only how you feel, what you experience, or the results of any diagnostic tests if asked (like x-rays or MRIs).
            5. If asked about medical terms, help with clues but do not reveal the diagnosis.
            6. Keep your responses precise and to the point, maximum 2 lines.
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


def play_text_tts(text, gender):
    """Perform TTS based on gender."""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id if gender == "female" else voices[1].id)
    engine.say(text)
    engine.runAndWait()

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

    # Create containers
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        if st.session_state.case_loaded and not st.session_state.asked_if_ready:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "A case has been selected. Ready to begin?"
            })
            st.session_state.asked_if_ready = True

        # Display chat messages
        for i, message in enumerate(st.session_state.chat_history):
            display_message(message, i)

    # Fixed bottom input section
    with st.container():
        if not st.session_state.ready_to_start:
            col1, col2 = st.columns([9, 1])
            with col1:
                user_input = st.text_input("Your response:", key="initial_input")
            with col2:
                send_button = st.button("Send", key="send_initial")
                
            if send_button and user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                if any(word in user_input.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'ready']):
                    st.session_state.ready_to_start = True
                    with st.spinner('Preparing case...'):
                        intro_response, _ = get_chatgroq_response("", is_introduction=True)
                        st.session_state.chat_history.append({"role": "assistant", "content": intro_response})
                        st.session_state.correct_diagnosis, _ = get_chatgroq_response("", is_diagnosis=True)
                    
                    response_text = "Let's begin!"
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                else:
                    response_text = "Let me know when you're ready to start."
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.experimental_rerun()
        else:
            col1, col2, col3 = st.columns([8, 1, 1])
            with col1:
                user_input = st.text_input("Your message:", key="chat_input", placeholder="Type your message here...")
            with col2:
                send_button = st.button("Send", key="send_chat")
            with col3:
                mic_button = st.button("🎤", key="mic_button")

            if send_button and user_input:
                # Process text input
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Display sentiment label
                interaction_style = analyze_sentiment(user_input)
                st.write(f"Interaction Style: {interaction_style}")

                with st.spinner('Thinking...'):
                    response, response_time = get_chatgroq_response(user_input)

                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.caption(f"Response time: {response_time:.2f} seconds")

                # Suggest a follow-up question
                context_prompt = PromptTemplate(
                    template="Based on the patient's last response: '{last_response}', suggest a follow-up question a physiotherapist might ask that a patient will understand. Make note the suggested question should be a ONE line question only.",
                    input_variables=["last_response"]
                )
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
                context_llm_chain = LLMChain(llm=llm, prompt=context_prompt)
                follow_up_question = context_llm_chain.run({"last_response": response})
                st.write("Suggested Follow-Up Question:", follow_up_question)

            if mic_button:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.write("Listening...")
                    audio = recognizer.listen(source)

                try:
                    user_input = recognizer.recognize_google(audio)
                    st.write(f"Recognized: {user_input}")

                    # Display sentiment label
                    interaction_style = analyze_sentiment(user_input)
                    st.write(f"Interaction Style: {interaction_style}")

                    st.session_state.chat_history.append({"role": "user", "content": user_input})

                    with st.spinner('Thinking...'):
                        response, response_time = get_chatgroq_response(user_input)

                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.caption(f"Response time: {response_time:.2f} seconds")

                    # Suggest a follow-up question
                    context_prompt = PromptTemplate(
                        template="Based on the patient's last response: '{last_response}', suggest a follow-up question a physiotherapist might ask that a patient will understand. Make note the suggested question should be a ONE line question only.",
                        input_variables=["last_response"]
                    )
                    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
                    context_llm_chain = LLMChain(llm=llm, prompt=context_prompt)
                    follow_up_question = context_llm_chain.run({"last_response": response})
                    st.write("Suggested Follow-Up Question:", follow_up_question)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()