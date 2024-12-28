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

# Set page config
st.set_page_config(page_title="ChatGroq PDF Chatbot Demo", layout="wide")

# Hardcoded API key
GROQ_API_KEY = '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'

# Initialize session state
if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = False
    st.session_state.vectors = None
    st.session_state.chat_history = []

def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def get_chatgroq_response(user_input):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        If the answer is not in the context, say "I don't have enough information to answer that question."
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
    st.title("ChatGroq PDF Chatbot Demo")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if not st.session_state.processed_pdf:
            with st.spinner('Processing PDF... This may take a few minutes.'):
                st.session_state.vectors = process_pdf(uploaded_file)
                st.session_state.processed_pdf = True
            st.success("PDF processed successfully!")

        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        user_input = st.chat_input("Ask a question about the PDF content:")

        if user_input:
            # Display user message
            st.chat_message("user").markdown(user_input)
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.spinner('Thinking...'):
                response, response_time = get_chatgroq_response(user_input)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
                st.caption(f"Response time: {response_time:.2f} seconds")

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    else:
        st.info("Please upload a PDF file to begin the conversation.")

if __name__ == "__main__":
    main()