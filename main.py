from fastapi import FastAPI, HTTPException
from typing import List
import fitz  # PyMuPDF
import re
import os

app = FastAPI()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting text from the PDF: {e}")
    
    return text

def preprocess_text(text: str) -> list:
    """Preprocesses the text to extract words."""
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def get_suggestions_from_text(word: str, text: str) -> list:
    """Generates suggestions from the text based on the input word."""
    words = preprocess_text(text)
    suggestions = [w for w in words if w.startswith(word) and w != word]
    return suggestions

# Path to your PDF file
PDF_PATH = "pdf/aopdf.pdf"

# Extract text from the PDF once at startup
try:
    pdf_text = extract_text_from_pdf(PDF_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"PDF file not found: {e}")
except RuntimeError as e:
    raise RuntimeError(f"Error loading PDF: {e}")

@app.get("/suggestions", response_model=List[str])
async def get_suggestions(word: str):
    """Provides suggestions based on the input word."""
    if not word:
        raise HTTPException(status_code=400, detail="Word query parameter is required")

    # Get suggestions from the extracted text
    suggestions = get_suggestions_from_text(word, pdf_text)
    
    return suggestions

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
