from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict
import fitz  # PyMuPDF
import re
import os
from collections import Counter
from pydantic import BaseModel
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pydantic import BaseModel



app = FastAPI()

class UserInputRequest(BaseModel):
    word: str
    next_word: str

class ParagraphRequest(BaseModel):
    paragraph: str

# Path to your PDF file
PDF_PATH = "aopdf.pdf"

# Initialize SQLite database
DATABASE_PATH = "user_inputs.db"

def init_db():
    """Initializes the database with a table for storing user inputs and outputs."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_input(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            next_word TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def store_user_input(word: str, next_word: str):
    """Stores the user's input and the next word in the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_input (word, next_word)
        VALUES (?, ?)
    ''', (word, next_word))
    conn.commit()
    conn.close()

def retrieve_user_inputs() -> List[Dict[str, str]]:
    """Retrieves all user inputs and their corresponding next words."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT word, next_word FROM user_input')
    rows = cursor.fetchall()
    conn.close()
    return [{"word": row[0], "next_word": row[1]} for row in rows]

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

def calculate_character_similarity(word1: str, word2: str) -> float:
    """Calculates the percentage of character similarity between two words."""
    matches = sum(1 for char1, char2 in zip(word1, word2) if char1 == char2)
    max_length = max(len(word1), len(word2))
    if max_length == 0:
        return 0.0
    return (matches / max_length) * 100

def check_word_matches(word: str, text_set: set) -> List[str]:
    """Checks each word for matching characters with words in the text."""
    matching_words = []
    for text_word in text_set:
        similarity = calculate_character_similarity(word.lower(), text_word.lower())
        if similarity > 50:  # Only consider similarities greater than 50%
            matching_words.append(text_word)
    
    # Remove duplicates and sort words
    matching_words = list(set(matching_words))
    matching_words.sort()
    
    return matching_words

def get_suggestions_from_text(word: str, text: str) -> List[str]:
    """Gets suggestions of words from the text that are similar to the input word."""
    suggestions = []
    for text_word in text.split():
        if text_word.lower().startswith(word.lower()):
            suggestions.append(text_word)
    return suggestions

def check_words_in_text(words: List[str], text_set: set) -> Dict[str, List[str]]:
    """Checks which words exist in the text set."""
    results = {"exists": [], "not_exists": []}
    for word in words:
        if word in text_set:
            results["exists"].append(word)
        else:
            results["not_exists"].append(word)
    return results

def find_next_words(word: str, text: str) -> List[str]:
    """Finds all words that follow the input word in the text and returns them sorted by frequency."""
    pattern = re.compile(r'\b' + re.escape(word) + r'\b\s+(\w+)', re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No words found after '{word}'")
    
    # Count the frequency of each next word
    word_counts = Counter(matches)
    
    # Sort words by frequency in descending order
    sorted_words = [word for word, count in word_counts.most_common()]
    
    return sorted_words

# Extract text from the PDF once at startup
try:
    pdf_text = extract_text_from_pdf(PDF_PATH)
    pdf_text_set = preprocess_text(pdf_text)  # Convert to set for fast lookup
except FileNotFoundError as e:
    raise RuntimeError(f"PDF file not found: {e}")
except RuntimeError as e:
    raise RuntimeError(f"Error loading PDF: {e}")

model, vectorizer = None, None

def train_ml_model():
    """Trains a machine learning model on the stored user inputs."""
    global model, vectorizer

    user_inputs = retrieve_user_inputs()
    if not user_inputs:
        return None, None

    # Prepare the data
    X = [ui['word'] for ui in user_inputs]
    y = [ui['next_word'] for ui in user_inputs]

    # Vectorize the input words with bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X)
    X_vectorized = vectorizer.transform(X)

    # Train a random forest classifier model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_vectorized, y)

    return model, vectorizer

# Train the model at startup
model, vectorizer = train_ml_model()

@app.get("/suggestions", response_model=List[str])
async def get_suggestions(word: str):
    """Provides suggestions based on the input word."""
    if not word:
        raise HTTPException(status_code=400, detail="Word query parameter is required")

    # Get suggestions from the extracted text
    suggestions = get_suggestions_from_text(word, pdf_text)
    
    return suggestions

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

@app.post("/check_word_matches", response_model=List[str])
async def check_word_matches_endpoint(request: ParagraphRequest):
    """Checks each word for matching characters with words in the PDF text and returns similar words."""
    word = request.paragraph.strip().lower()  # Treat the input word as the only word
    if not word:
        raise HTTPException(status_code=400, detail="Word body is required")

    # Calculate similarity for each word in the PDF text
    results = check_word_matches(word, pdf_text_set)

    return results

@app.get("/find_next_word", response_model=List[str])
async def find_next_word(word: str):
    """Finds and returns all words that follow the input word in the PDF."""
    if not word:
        raise HTTPException(status_code=400, detail="Word query parameter is required")

    # Find next words from the extracted text
    next_words = find_next_words(word, pdf_text)
    
    return next_words
@app.post("/store_user_input")
async def store_user_input_endpoint(request: ParagraphRequest):
    """Stores word pairs from a paragraph and retrains the ML model."""
    paragraph = request.paragraph
    
    if not paragraph:
        raise HTTPException(status_code=400, detail="Paragraph body is required")

    words = re.findall(r'\b\w+\b', paragraph.lower())
    
    if len(words) < 2:
        raise HTTPException(status_code=400, detail="Paragraph must contain at least two words")
    
    # Store each word and its following word
    for i in range(len(words) - 1):
        word = words[i]
        next_word = words[i + 1]
        store_user_input(word, next_word)
    
    # Retrain the ML model with the new data
    global model, vectorizer
    model, vectorizer = train_ml_model()
    
    return {"message": "User inputs stored and model retrained successfully"}

    
    return {"message": "User input stored successfully"}

@app.get("/predict_next_word", response_model=List[str])
async def predict_next_word(word: str):
    """Predicts the next word based on the input word using the trained ML model."""
    if not word:
        raise HTTPException(status_code=400, detail="Word query parameter is required")

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model has not been trained yet")

    # Check if the word exists in the training data
    if word not in [ui['word'] for ui in retrieve_user_inputs()]:
        return ["Learning required: The word is not in the training data."]

    X_vectorized = vectorizer.transform([word])

    # Get prediction probabilities for all classes
    prediction_probs = model.predict_proba(X_vectorized)

    # Get top predictions and their probabilities
    top_indices = prediction_probs[0].argsort()[::-1]
    top_predictions = [model.classes_[i] for i in top_indices if prediction_probs[0][i] > 0.01]  # Filter out very low probabilities

    # Sort predictions by frequency from training data
    word_counts = Counter(ui['next_word'] for ui in retrieve_user_inputs() if ui['word'] == word)
    sorted_predictions = sorted(top_predictions, key=lambda w: word_counts[w], reverse=True)

    return sorted_predictions


# Initialize the database
init_db()
