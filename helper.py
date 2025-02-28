import os
import re
import string
import unicodedata
import pickle
import numpy as np
import nltk
import pandas as pd
from googletrans import Translator  # Now uses synchronous 3.1.0a0
from logger import logging

# ====== SETUP NLTK PATHS ====== #
nltk.data.path.append('static/model/nltk_data')
from nltk.stem import RSLPStemmer

# ====== INITIALIZE COMPONENTS ====== #
translator = Translator()
stemmer = RSLPStemmer()

# ====== LOAD RESOURCES ====== #
with open('static/model/logistic_regression_Latest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('static/model/portuguese_vocabulary.txt', 'r', encoding='utf-8') as f:
    tokens = f.read().splitlines()

with open('static/model/nltk_data/corpora/stopwords/portuguese', 'r', encoding='utf-8') as f:
    pt_stopwords = set(f.read().splitlines())

# ====== TRANSLATION FUNCTION ====== #
def translate_to_portuguese(text):
    """Translate English text to Portuguese"""
    try:
        translated = translator.translate(text, src='en', dest='pt').text
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        logging.error(f"Translation error: {e}")
        return text  # Fallback to original text

# ====== TEXT PREPROCESSING FUNCTION ====== #
def preprocess_text_pt(text):
    """Clean and prepare Portuguese text for analysis"""
    if pd.isna(text):
        return ''

    # Text normalization
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)  # Remove URLs/mentions
    text = re.sub(r'[^\w\sáàâãéêíóôõúç]', '', text)  # Keep Portuguese chars
    text = ''.join(c for c in text if unicodedata.category(c) != 'So')  # Remove emojis
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")  # Remove accents
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers

    # Language processing
    text = ' '.join([word for word in text.split() if word not in pt_stopwords])  # Remove stopwords
    text = ' '.join([stemmer.stem(word) for word in text.split()])  # Stemming

    return text

# ====== VECTORIZATION FUNCTION ====== #
def vectorizer(text, vocabulary):
    """Convert text to binary feature vector"""
    vectorized = np.zeros(len(vocabulary), dtype=np.float32)
    words = set(text.split())
    for i, word in enumerate(vocabulary):
        vectorized[i] = 1 if word in words else 0
    return vectorized.reshape(1, -1)

# ====== FULL PREDICTION PIPELINE ====== #
def analyze_text(text,language='en'):
    """
    Complete processing pipeline for English text:
    1. Translate to Portuguese if language is English
    2. Preprocess text
    3. Vectorize features
    4. Predict sentiment
    """

    # Translation phase (only if the language is English)
    if language == 'en':
        pt_text = translate_to_portuguese(text)
        logging.info(f"Translated Portuguese text: {pt_text}")
    else:
        pt_text = text  # No translation needed for Portuguese input
        logging.info(f"Using input text as is (Portuguese): {pt_text}")

    # Preprocessing phase
    cleaned_text = preprocess_text_pt(pt_text)
    logging.info(f"Preprocessed text: {cleaned_text}")  # Print cleaned text

    # Vectorization phase
    vectorized = vectorizer(cleaned_text, tokens)

    # Prediction phase
    prediction = model.predict(vectorized)[0]
    return "positive" if prediction == 1 else "negative"