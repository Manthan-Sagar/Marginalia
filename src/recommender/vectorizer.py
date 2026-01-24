from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import scipy.sparse
import json
import os

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def build_vectorizer():
    return TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.85
    )

def fit_and_save_vectorizer(df):
    """
    Fits the vectorizer on the 'tags' column and saves artifacts.
    """
    print("Fitting vectorizer...")
    vectorizer = build_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['tags'])
    
    print(f"Matrix shape: {tfidf_matrix.shape}")
    
    # Save Artifacts
    vec_path = os.path.join(ARTIFACTS_DIR, 'vectorizer.pkl')
    matrix_path = os.path.join(ARTIFACTS_DIR, 'book_vectors.npz')
    index_path = os.path.join(ARTIFACTS_DIR, 'id_to_index.json') # Saving inside models too for consistency
    
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    scipy.sparse.save_npz(matrix_path, tfidf_matrix)
    
    # Save ID mapping (Title -> Index)
    # Using title as ID for now.
    id_to_index = {title: idx for idx, title in enumerate(df['title'])}
    with open(index_path, 'w') as f:
        json.dump(id_to_index, f)
        
    print(f"Artifacts saved to {ARTIFACTS_DIR}")
    return vectorizer, tfidf_matrix

def load_vectorizer(artifacts_dir=ARTIFACTS_DIR):
    """
    Loads the vectorizer and matrix from disk.
    """
    vec_path = os.path.join(artifacts_dir, 'vectorizer.pkl')
    matrix_path = os.path.join(artifacts_dir, 'book_vectors.npz')
    
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    matrix = scipy.sparse.load_npz(matrix_path)
    
    return vectorizer, matrix
