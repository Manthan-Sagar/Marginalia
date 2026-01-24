import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# Ensure resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

def clean_description(text):
    """
    Cleans description text:
    - Lowercase
    - Remove punctuation
    - Remove stopwords (keeping negations)
    - Lemmatize
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords (custom list to keep negations)
    stop_words = set(stopwords.words('english'))
    negations = {'no', 'not', 'nor', 'neither', 'never', 'none'}
    stop_words = stop_words - negations
    
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    return " ".join(lemmatized)

def preprocess_data(df):
    """
    Applies all preprocessing steps to the dataframe.
    """
    print("Preprocessing data...")
    
    # 1. Clean Category Names: Replace " | " with " "
    if 'category_names' in df.columns:
        df['category_names'] = df['category_names'].str.replace(' | ', ' ', regex=False)
    
    # 2. Lowercase Metadata
    cols_to_lower = ['author_names', 'category_names', 'title']
    for col in cols_to_lower:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
            
    # 3. Clean Description (Custom Logic)
    if 'description' in df.columns:
        df['cleaned_description'] = df['description'].apply(clean_description)
    
    # 4. Create 'tags' column
    # fillna('') is crucial to avoid NaN results
    df['tags'] = (
        df['cleaned_description'].fillna('') + ' ' + 
        df['author_names'].fillna('') + ' ' + 
        df['category_names'].fillna('') + ' ' + 
        df['title'].fillna('')
    )
    
    # Clean up extra spaces
    df['tags'] = df['tags'].apply(lambda x: " ".join(x.split()))
    
    print("Preprocessing complete.")
    return df
