from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def search_books(query, vectorizer, book_vectors, df, top_n=5, filters=None):
    """
    Searches for books similar to the query using cosine similarity.
    Applies filters (author) and sorts final results by average rating.
    """
    # 1. Transform query
    query_vector = vectorizer.transform([query])
    
    # 2. Handle Filtering
    search_indices = df.index  # Default: search everyone
    
    if filters:
        # --- Author Filter ---
        if 'author' in filters and filters['author']:
            target_author = filters['author'].lower().replace(" ", "")
            
            def match_author(book_authors):
                if pd.isna(book_authors): return False
                # Normalize book authors: remove spaces, lowercase
                # User said: "austinclinton georgefernald" (clumped)
                # But to be safe we'll assume we search for our clumped target IN the book string
                return target_author in str(book_authors).lower()
                
            author_mask = df['author_names'].apply(match_author)
            
            if author_mask.sum() > 0:
                print(f"Filtering by Author: '{filters['author']}' (Found {author_mask.sum()} books)")
                search_indices = search_indices[author_mask]
            else:
                print(f"Warning: Author '{filters['author']}' not found. Searching full dataset.")

    # Slice matrix
    current_vectors = book_vectors[search_indices]
    
    # 3. Compute Cosine Similarity
    if current_vectors.shape[0] == 0:
        print("No books match filters.")
        return []

    scores = cosine_similarity(query_vector, current_vectors)
    
    # 4. Get Top N candidates (Fetch extra to allow for deduplication)
    candidate_limit = top_n * 4
    top_local_indices = scores[0].argsort()[-candidate_limit:][::-1]
    
    # 5. Collect and Deduplicate Results
    results = []
    seen_books = [] # List of {'title': str, 'authors': set}
    
    print(f"\nQuery: '{query}' (Filters: {filters})")
    
    for local_idx in top_local_indices:
        if len(results) >= top_n:
            break
            
        score = scores[0][local_idx]
        original_idx = search_indices[local_idx]
        book = df.iloc[original_idx]
        
        # Normalize for comparison
        # Aggressive normalization to catch "Title : Subtitle" or "Title (Edition)"
        title_raw = str(book['title'])
        title_base = title_raw.split('(')[0].split(':')[0].strip().lower()
        
        authors_str = str(book.get('author_names', ''))
        authors_set = set(authors_str.lower().split()) # Split by space as per user spec
        
        is_duplicate = False
        for seen in seen_books:
            # Check Title Match (Base title comparison)
            if seen['title_base'] == title_base:
                # Check Author Overlap
                # If ANY author overlaps, treat as same book
                if (not seen['authors'] and not authors_set) or (seen['authors'] & authors_set):
                    is_duplicate = True
                    break
        
        if is_duplicate:
            continue
            
        # Add to results
        seen_books.append({'title_base': title_base, 'authors': authors_set})
        
        results.append({
            'title': book['title'],
            'score': score,
            'authors': book.get('author_names', 'N/A'),
            'rating': book.get('rating-avg', 0),
            'description': str(book.get('description', 'N/A'))[:100] + "..."
        })
        print(f"Score: {score:.4f} | Rating: {book.get('rating-avg', 0)} | {book['title']} | {book.get('author_names', '')}")
        
    return results
