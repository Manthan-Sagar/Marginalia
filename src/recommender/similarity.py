from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_books(query, vectorizer, book_vectors, df, top_n=5):
    """
    Searches for books similar to the query using cosine similarity.
    """
    # 1. Transform query
    query_vector = vectorizer.transform([query])
    
    # 2. Compute Cosine Similarity
    scores = cosine_similarity(query_vector, book_vectors)
    
    # 3. Get Top N indices
    top_indices = scores[0].argsort()[-top_n:][::-1]
    
    # 4. Return results
    results = []
    print(f"\nQuery: '{query}'")
    for idx in top_indices:
        score = scores[0][idx]
        book = df.iloc[idx]
        results.append({
            'title': book['title'],
            'score': score,
            'authors': book.get('author_names', 'N/A'),
            'description': book.get('description', 'N/A')[:100] + "..."
        })
        print(f"Score: {score:.4f} | {book['title']}")
        
    return results
