# /python/embedding_worker.py

"""
ONNX Embedding Worker for EmbeddingGemma model
- Implements ONNX Runtime for efficient embedding generation.
- Supports token counting and vector embedding generation.
- Handles batch embedding requests via stdin/stdout JSON communication.
- Applies EmbeddingGemma optimized formatting rules before encoding.
"""

import sys
import json
import logging
from typing import Optional
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Configure logging to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[EmbedWorker] %(message)s')
logger = logging.getLogger()

# Constants
MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"
MAX_LENGTH = 2048 # Gemma 300m supports up to 2048 tokens

# Task Definitions
TASK_PROMPTS = {
    "retrieval": "search result",
    "question_answering": "question answering",
    "fact_checking": "fact checking",
    "classification": "classification",
    "clustering": "clustering",
    "similarity": "sentence similarity",
    "code_retrieval": "code retrieval",
}

class ONNXEmbedder:
    def __init__(self):
        logger.info(f"Initializing ONNX Session for {MODEL_ID}...")
        
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # 2. Download/Locate Model
        model_path = hf_hub_download(
            repo_id=MODEL_ID, 
            subfolder="onnx", 
            filename="model_quantized.onnx"
        )
        hf_hub_download(
            repo_id=MODEL_ID, 
            subfolder="onnx", 
            filename="model_quantized.onnx_data"
        )

        # 3. Low Memory Configuration
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.intra_op_num_threads = 2
        sess_opt.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path, 
            sess_opt, 
            providers=['CPUExecutionProvider']
        )
        logger.info("ONNX Model loaded successfully.")

    def count_tokens(self, texts):
        """Returns the exact token count for a list of strings"""
        # We don't truncate here because the user wants to know the REAL length 
        # to decide where to split.
        encoded = self.tokenizer(texts, add_special_tokens=True, return_attention_mask=False)
        return [len(ids) for ids in encoded['input_ids']]

    def encode_batch(self, texts):
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="np"
        )
        
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        outputs = self.session.run(None, onnx_inputs)
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return self.normalize(embeddings)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(token_embeddings.shape[-1], axis=-1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def normalize(self, v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

def format_text(text: str, title: Optional[str] = None, task_type: Optional[str] = None) -> str:
    if task_type:
        prompt = TASK_PROMPTS.get(task_type, "search result")
        return f"task: {prompt} | query: {text}"
    if title:
        return f"title: {title} | text: {text}"
    return f"title: none | text: {text}"

def main():
    try:
        embedder = ONNXEmbedder()
    except Exception as e:
        logger.error(f"Fatal Init Error: {e}")
        sys.exit(1)

    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            
            req = json.loads(line)
            raw_texts = req.get("texts", [])
            command = req.get("command", "embed") # Default to embed
            
            if not raw_texts:
                sys.stdout.write(json.dumps({"error": "No texts provided"}) + "\n")
                sys.stdout.flush()
                continue

            if command == "count_tokens":
                # Just return integers
                counts = embedder.count_tokens(raw_texts)
                resp = {"token_counts": counts}
                
            else:
                # Default: Embedding
                title = req.get("title")
                task_type = req.get("task_type")
                formatted = [format_text(t, title, task_type) for t in raw_texts]
                vectors = embedder.encode_batch(formatted)
                resp = {"vectors": vectors.tolist()}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"Processing Error: {e}")
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()