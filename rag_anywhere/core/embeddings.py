# rag_anywhere/core/embeddings.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class EmbeddingGemmaProvider:
    """
    EmbeddingGemma-300m embedding provider
    - 768 dimensions
    - 2048 token context
    - Multi-lingual support
    """
    
    def __init__(self, model_name: str = "google/embeddinggemma-300m", device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading EmbeddingGemma on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        self.max_length = 2048
        self.dimension = 768
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token or mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def embed_single(self, text: str) -> np.ndarray:
        """Convenience method for single text"""
        return self.embed([text])[0]
