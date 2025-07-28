import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load multilingual sentence transformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def compute_heading_score(text):
    """
    Returns a heuristic score for whether text seems like a heading.
    Higher score = more heading-like.
    """
    score = 0
    clean_text = text.strip()
    
    if not clean_text:
        return -5 # Heavily penalize empty lines

    # --- Enhanced Logic ---
    # Penalize if the text is mostly non-alphanumeric characters (e.g., "----------")
    # Counts characters that are NOT letters, numbers, or whitespace
    non_alnum_count = len(re.findall(r'[^\w\s]', clean_text))
    if non_alnum_count > len(clean_text) * 0.6:
        score -= 3

    # Reward short lines, common for headings
    if len(clean_text.split()) <= 12:
        score += 1
    
    # Reward uppercase text
    if clean_text.isupper():
        score += 1
        
    # Reward lines that don't end with a period
    if not clean_text.endswith("."):
        score += 1
        
    # Reward numbered sections like "1.1 Introduction"
    if re.match(r"^\d+(\.\d+)*\s", clean_text):
        score += 2
        
    # Penalize patterns that look like full names
    if re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", clean_text):
        score -= 1

    return score

def get_heading_score(text, previous_texts, next_texts, k=3):
    """
    Computes semantic similarity between this line and nearby lines
    using SentenceTransformer embeddings.
    Lower similarity suggests it's a distinct heading.
    """
    try:
        current_embedding = model.encode(text, convert_to_tensor=True)

        nearby_texts = previous_texts[-k:] + next_texts[:k]
        if not nearby_texts:
            return 0.0

        nearby_embeddings = model.encode(nearby_texts, convert_to_tensor=True)
        # Calculate cosine similarity
        similarities = util.cos_sim(current_embedding, nearby_embeddings)[0]
        avg_similarity = similarities.mean().item()
        
        # Heading score is the inverse of similarity (1 - sim)
        heading_score = 1.0 - avg_similarity
        return round(heading_score, 4)

    except Exception as e:
        print(f"[NLP Warning] Failed heading score for '{text[:30]}...': {e}")
        return 0.0