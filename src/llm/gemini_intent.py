import os
import json
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Client
API_KEY = os.getenv("GEMINI_API_KEY")

def extract_intent(user_text):
    """
    Uses Gemini to extract structured intent from user text.
    Returns a dictionary with themes, tone, preferred_genres, excluded_genres.
    """
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found.")
        return {
            "themes": [],
            "tone": [],
            "preferred_genres": [],
            "excluded_genres": []
        }

    try:
        # Retry config for rate limits
        # Using a simple retry loop for 429 errors
        from google.api_core import exceptions
        import time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = genai.Client(api_key=API_KEY)
                
                prompt = f"""
You are extracting search intent for a book recommendation system.

User Text:
"{user_text}"

Return ONLY valid JSON with the following schema:
{{
  "themes": ["theme1", "theme2"],
  "tone": ["tone1", "tone2"],
  "preferred_genres": ["genre1"],
  "excluded_genres": ["genre2"]
}}

STRICT RULES (must follow):
1. Output ONLY valid JSON. No markdown, no explanations, no extra text.
2. Each theme or tone MUST be a short, normalized concept (1–3 words max).
   - GOOD: "happiness", "meaning of life", "self discovery"
   - BAD: "book about happiness and life", "a story that makes you happy"
3. Do NOT use full sentences.
4. Normalize vague language into standard concepts.
5. Genre depth must be at most 2 words (e.g., "literary fiction", "self help").
6. If tone is unclear, infer a reasonable tone instead of leaving it empty.
7. If no excluded genres are mentioned, return an empty list.

Return JSON only.
"""
                # Using gemini-flash-latest (Verified working)
                response = client.models.generate_content(
                    model='gemini-flash-latest', 
                    contents=prompt
                )
                
                text = response.text.strip()
                
                # Clean markdown if present
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                    
                intent = json.loads(text)
                return intent

            except exceptions.ResourceExhausted:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Quota exceeded. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise # Re-raise to fall through to general exception handler
            except Exception:
                raise # Re-raise other errors immediately

    except Exception as e:
        print(f"Warning: Gemini extraction failed: {e}")
        # FALLBACK: Use raw user text as a 'theme' so search still runs
        return {
            "themes": [user_text],
            "tone": [],
            "preferred_genres": [],
            "excluded_genres": []
        }

def intent_to_query_text(intent):
    """
    Converts the intent dictionary into a single string for vector search.
    """
    terms = []
    
    # Flatten lists
    terms.extend(intent.get("themes", []))
    terms.extend(intent.get("tone", []))
    terms.extend(intent.get("preferred_genres", []))
    
    # Join into single string
    return " ".join(terms)

if __name__ == "__main__":
    # Test block
    print("Testing Gemini Intent Extraction (v2 SDK)...")
    if not API_KEY:
        print("⚠️  No API Key found. Please check your .env file.")
    else:
        test_text = "I want a dark fantasy about a thief who steals from gods."
        print(f"\nUser Text: {test_text}")
        
        intent = extract_intent(test_text)
        print("\nExtracted Intent:")
        print(json.dumps(intent, indent=2))
        
        query = intent_to_query_text(intent)
        print(f"\nQuery String: {query}")
