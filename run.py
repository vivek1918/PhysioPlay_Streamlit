from extract_pdf_text import extract_text_from_pdf
from generate_embeddings import get_pdf_embedding
from faiss_index import create_faiss_index, add_to_index
from random_selector import select_random_pdf
import os

def main():
    pdf_folder = './data/'
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    # Extract text and generate embeddings
    pdf_texts = [extract_text_from_pdf(pdf) for pdf in pdf_files]
    pdf_embeddings = [get_pdf_embedding(text) for text in pdf_texts]

    # Create FAISS index and add embeddings
    gpu_index = create_faiss_index()
    add_to_index(gpu_index, pdf_embeddings)

    # Randomly select a PDF
    random_pdf = select_random_pdf(gpu_index, pdf_files)
    print(f"Randomly selected PDF: {random_pdf}")

if __name__ == "__main__":
    main()
