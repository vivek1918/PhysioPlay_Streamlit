import random

def select_random_pdf(index, pdf_paths):
    num_pdfs = index.ntotal
    random_idx = random.randint(0, num_pdfs - 1)
    return pdf_paths[random_idx]
