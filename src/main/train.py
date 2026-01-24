import pandas as pd
import os
import sys
import gc

# Add src to path
sys.path.append(os.path.dirname(__file__))

from recommender import preprocessing, vectorizer

def main():
    print("--- Training on Full Dataset ---")
    
    # Path to full dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'processed_book_data_kaggle.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    # OPTIMIZATION: Load only necessary columns
    use_cols = ['title', 'description', 'author_names', 'category_names', 'tags']
    
    try:
        # Load columns that exist
        # We handle 'tags' if it doesn't exist by creating it in preprocessing
        df = pd.read_csv(data_path, usecols=lambda c: c in use_cols or c in ['title', 'description', 'author_names', 'category_names'])
        print(f"Loaded {len(df)} rows.")
        
    except Exception as e:
        print(f"Error loading full CSV: {e}")
        return

    # Preprocess
    print("Preprocessing/Feature Engineering...")
    df = preprocessing.preprocess_data(df)
    
    # Garbage collect
    gc.collect()

    # Fit and Save
    print("Fitting Vectorizer (this may take a while)...")
    try:
        vectorizer.fit_and_save_vectorizer(df)
        print("Training Complete!")
    except MemoryError:
        print("CRITICAL: Out of Memory during vectorization.")
        print("Try reducing max_features in vectorizer.py")

if __name__ == "__main__":
    main()
