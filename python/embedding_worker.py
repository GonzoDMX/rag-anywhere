# /python/embedding_worker.py

"""
Embedding Worker for EmbeddingGemma model
- Handles batch embedding requests via stdin/stdout JSON communication.
- Applies EmbeddingGemma optimized formatting rules before encoding.
"""

import sys
import json
import logging
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging to stderr (Stdout is for JSON only)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[EmbedWorker] %(message)s')
logger = logging.getLogger()

# Task definitions specific to EmbeddingGemma
TASK_PROMPTS = {
    "retrieval_document": "none",
    "retrieval_query": "search result",           # Optimized for document retrieval queries
    "question_answering": "question answering",   # Optimized for QA tasks
    "fact_verification": "fact checking",         # Optimized to verify factual claims against provided evidence
    "classification": "classification",           # Optimized to classify texts according to preset labels
    "clustering": "clustering",                   # Optimized to cluster texts based on their similarities
    "semantic_similarity": "sentence similarity", # Optimized to assess text similarity. Not intended for retrieval use cases.
    "code_retrieval": "code retrieval",           # Optimized for retrieving code snippets based on natural language queries
}

def load_model():
    # In production, this might come from env var or CLI arg
    model_name = "google/embeddinggemma-300m"
    logger.info(f"Loading model: {model_name}...")
    try:
        # trust_remote_code=True is required for Gemma
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback for testing/dev without internet or massive download
        logger.warning("Attempting fallback to 'all-MiniLM-L6-v2'")
        return SentenceTransformer('all-MiniLM-L6-v2')

def format_text(text: str, title: Optional[str] = None, task_type: Optional[str] = None) -> str:
    """Applies EmbeddingGemma specific formatting rules"""
    
    # CASE 1: Document Retrieval Mode
    if task_type == "retrieval_document":
        header = f"title: none"
        if title:
            header = f"title: {title}"
        return f"{header} | text: {text}"

    # CASE 2: Optimized Task Retrieval Mode
    if task_type:
        header = TASK_PROMPTS.get(task_type, "search result")
        return f"task: {header} | query: {text}"
    
    # Default fallback (just the text)
    return text

def main():
    model = load_model()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break # EOF
            
            req = json.loads(line)
            raw_texts = req.get("texts", [])
            title = req.get("title")
            task_type = req.get("task_type")
            
            if not raw_texts:
                sys.stdout.write(json.dumps({"error": "No texts provided"}) + "\n")
                sys.stdout.flush()
                continue

            # 1. Pre-process / Format all texts in the batch
            formatted_texts = [
                format_text(t, title, task_type) 
                for t in raw_texts
            ]

            # 2. Encode Batch
            # normalize_embeddings=True is CRITICAL for dot-product to equal cosine similarity
            embeddings: np.ndarray = model.encode( 
                formatted_texts, 
                batch_size=32, 
                normalize_embeddings=True,
                convert_to_numpy=True
            ) 

            # 3. Serialize
            # Convert numpy float32 to standard python float list
            resp = {"vectors": embeddings.tolist()}
            
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"Processing Error: {e}")
            error_resp = {"error": str(e)}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()