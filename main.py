from fastapi import FastAPI, HTTPException
from typing import List, Dict
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

def preprocess_text(text: str) -> set:
    """Preprocesses the text to extract words and store them in a set for fast lookup."""
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)

def get_suggestions_from_text(word: str, text: str) -> List[str]:
    """Generates suggestions from the text based on the input word."""
    words = preprocess_text(text)
    suggestions = [w for w in words if w.startswith(word) and w != word]
    return suggestions

def check_words_in_text(words: List[str], text_set: set) -> Dict[str, List[str]]:
    """Checks which words are in the text and which are not."""
    results = {"exist": [], "notexist": []}
    for word in words:
        if word.lower() in text_set:
            results["exist"].append(word)
        else:
            results["notexist"].append(word)
    return results

# Path to your PDF file
PDF_PATH = "aopdf.pdf"

# Extract text from the PDF once at startup
try:
    pdf_text = extract_text_from_pdf(PDF_PATH)
    pdf_text_set = preprocess_text(pdf_text)  # Convert to set for fast lookup
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
from pydantic import BaseModel

class ParagraphRequest(BaseModel):
    paragraph: str

@app.post("/check_paragraph", response_model=Dict[str, List[str]])
async def check_paragraph(request: ParagraphRequest):
    """Checks each word in the paragraph and provides which words exist in the PDF text and which do not."""
    paragraph = request.paragraph
    if not paragraph:
        raise HTTPException(status_code=400, detail="Paragraph body is required")

    words = re.findall(r'\b\w+\b', paragraph.lower())
    results = check_words_in_text(words, pdf_text_set)
    
    return results


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
