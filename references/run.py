from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import random

from extract_pdf_text import extract_text_from_pdf  # Ensure this exists
from generate_embeddings import get_pdf_embedding  # Ensure this exists
from faiss_index import create_faiss_index, add_to_index, search_index  # Ensure these exist
from random_selector import select_random_pdf  # Ensure this exists

app = FastAPI()

class PDFPath(BaseModel):
    pdf_path: str

class SearchQuery(BaseModel):
    query: List[float]
    top_k: int = 5

# Create FAISS index only once at startup
gpu_index = create_faiss_index()

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Search API!"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to the server and save it to a specified folder.
    """
    pdf_folder = './data/'  # Ensure this folder exists or create it dynamically
    os.makedirs(pdf_folder, exist_ok=True)  # Create the folder if it doesn't exist

    file_location = os.path.join(pdf_folder, file.filename)
    try:
        with open(file_location, "wb") as pdf_file:
            pdf_file.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    return {"filename": file.filename, "file_location": file_location}

@app.post("/initialize_index/")
async def initialize_index(pdf_folder: str = './data/'):
    """
    Initialize FAISS index by extracting embeddings from all PDFs in the specified folder.
    """
    try:
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        if not pdf_files:
            raise HTTPException(status_code=404, detail="No PDF files found for indexing.")

        pdf_texts = [extract_text_from_pdf(pdf) for pdf in pdf_files]
        pdf_embeddings = [get_pdf_embedding(text) for text in pdf_texts]
        add_to_index(gpu_index, pdf_embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize index: {str(e)}")
    
    return {"message": "Index initialized and PDFs processed successfully!"}

@app.get("/random_pdf/")
async def get_random_pdf(pdf_folder: str = './data/'):
    """
    Get a random PDF from the specified folder.
    """
    try:
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        random_pdf = select_random_pdf(pdf_files)
        return {"random_pdf": random_pdf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error selecting random PDF: {str(e)}")

@app.post("/search_pdf/")
async def search_pdf(query: SearchQuery):
    """
    Search for PDFs in the FAISS index based on a query vector.
    """
    try:
        results = search_index(gpu_index, query.query, top_k=query.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Example of select_random_pdf function
def select_random_pdf(pdf_paths: List[str]) -> str:
    """
    Select a random PDF file from the list of provided PDF paths.
    """
    if not pdf_paths:
        raise ValueError("The list of PDF paths is empty.")
    return random.choice(pdf_paths)

@app.post("/process_random_pdf/")
async def process_random_pdf(pdf_folder: str = './data/'):
    """
    Process a random PDF from the specified folder and return its path.
    """
    try:
        # Step 1: Ensure the folder exists
        if not os.path.exists(pdf_folder):
            raise HTTPException(status_code=400, detail=f"PDF folder '{pdf_folder}' does not exist.")

        # Step 2: Get all PDFs in the folder
        pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

        # Step 3: Check if PDFs exist in the folder
        if not pdf_paths:
            raise HTTPException(status_code=404, detail="No PDF files found in the folder.")

        # Step 4: Select a random PDF (pass pdf_paths to the function)
        random_pdf = select_random_pdf(pdf_paths)  # Pass pdf_paths here

        # Additional logging for debugging
        print(f"Selected random PDF: {random_pdf}")

        return {"message": "Processed random PDF successfully!", "pdf": random_pdf}
    except HTTPException as http_ex:
        # Return the HTTPException raised
        return JSONResponse(status_code=http_ex.status_code, content={"message": http_ex.detail})
    except Exception as e:
        # Catch any unexpected exceptions and log them
        print(f"Error during processing random PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the random PDF: {str(e)}")

@app.get("/get_case_introduction/")
async def get_case_introduction(pdf_folder: str = './data/'):
    try:
        # Get all PDF files in the specified folder
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        if not pdf_files:
            raise HTTPException(status_code=404, detail="No PDF files found.")
        
        # Select a random PDF and extract its text
        random_pdf = select_random_pdf(pdf_files)
        extracted_text = extract_text_from_pdf(random_pdf)

        # Process the extracted text to create a concise introduction
        # Split the text into sentences or lines and select the first 1-2 lines
        lines = extracted_text.splitlines()
        case_intro = " ".join(line.strip() for line in lines if line.strip())[:200]  # Get up to 200 characters

        # Trim the case introduction to ensure it fits into 1-2 lines, while not exceeding 200 characters
        if len(case_intro) > 200:
            case_intro = case_intro[:197] + "..."  # Add ellipsis if trimmed

        return {"random_pdf": random_pdf, "case_introduction": case_intro}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class DiagnosisSubmission(BaseModel):
    diagnosis: str

@app.post("/submit_diagnosis/")
async def submit_diagnosis(diagnosis_submission: DiagnosisSubmission):
    """
    Submit a diagnosis to the server.
    """
    return {"message": "Diagnosis submitted successfully!", "diagnosis": diagnosis_submission.diagnosis}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
