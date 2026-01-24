import os
import sys
import pandas as pd

# Add src to path for imports
# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommender import vectorizer, similarity
from llm import gemini_intent

def main():
    print("Initializing Narrative Book Recommender...")
    
    # 1. Load Data
    # Pointing to the FULL dataset now
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'processed_book_data_kaggle.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path} (this may take a moment)...")
    # Optimize loading for inference: we only need title and description for display/mapping
    # We DO NOT need to load 1GB of 'description' strings into RAM if we just map by ID?
    # Actually similarity.py returns full rows, so we do need descriptions.
    df = pd.read_csv(data_path)
    
    # 2. Load Models
    try:
        vec, matrix = vectorizer.load_vectorizer()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run preprocessing and training first.")
        return

    print("\n--- Model Ready ---")
    print("Describe what you are looking for (or type 'exit' to quit).")
    
    while True:
        user_input = input("\nYour Request: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
        
        if not user_input:
            continue
            
        print("\nAnalyzing intent with Gemini...")
        
        # 3. Extract Intent
        intent = gemini_intent.extract_intent(user_input)
        
        # Format intent for display
        print(f"Themes: {intent.get('themes')}")
        print(f"Tone: {intent.get('tone')}")
        # Format intent for display
        print(f"Themes: {intent.get('themes')}")
        print(f"Tone: {intent.get('tone')}")
        
        # 4. Secondary Filter Input
        print("\n--- Optional Filters ---")
        
        filters = {}
        
        # Author Filter
        author_input = input("Filter by Author (name only, leave blank to skip): ").strip()
        if author_input:
            filters['author'] = author_input
            
        # Page Filter
        page_input = input("Filter by Page Count (e.g. 100-300, leave blank to skip): ").strip()
        if page_input:
            if '-' in page_input and page_input.replace('-', '').isdigit():
                print("⚠️  Warning: Page count data is not available in the current dataset. Page filter ignored.")
                # filters['pages'] = page_input
            else:
                 print(f"⚠️  Ignoring invalid page format: {page_input}")

        # 5. Convert to Query
        query_str = gemini_intent.intent_to_query_text(intent)
        
        # 6. Search
        print("\nTop Recommendations:")
        similarity.search_books(
            query=query_str,
            vectorizer=vec,
            book_vectors=matrix,
            df=df,
            top_n=5,
            filters=filters
        )

if __name__ == "__main__":
    main()
