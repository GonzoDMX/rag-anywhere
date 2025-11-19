# rag_anywhere/core/embeddings/factory.py

from typing import Dict, Any

from .base import EmbeddingProvider
from .providers import (
    EmbeddingGemmaProvider,
    OpenAIEmbeddingProvider,
)


class EmbeddingProviderFactory:
    """
    Factory for creating embedding providers.
    Handles dimension compatibility automatically.
    """
    
    STANDARD_DIMENSION = 768
    
    @staticmethod
    def create_provider(config: Dict[str, Any]) -> EmbeddingProvider:
        provider_type = config.get('provider', 'embeddinggemma')
        
        if provider_type == 'embeddinggemma':
            provider = EmbeddingGemmaProvider(
                model_name=config.get('model', 'google/embeddinggemma-300m')
            )
        
        elif provider_type == 'openai':
            provider = OpenAIEmbeddingProvider(
                api_key=config['api_key'],
                model=config.get('model', 'text-embedding-3-small')
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        # Validate dimension
        EmbeddingProviderFactory.validate_provider(provider)
        return provider
    
    @staticmethod
    def validate_provider(provider: EmbeddingProvider):
        """Ensure provider meets requirements"""
        if provider.dimension != EmbeddingProviderFactory.STANDARD_DIMENSION:
            raise ValueError(
                f"Provider {provider.name} outputs {provider.dimension} dimensions. "
                f"RAG Anywhere requires {EmbeddingProviderFactory.STANDARD_DIMENSION}-dimensional embeddings."
            )
    
    @staticmethod
    def list_providers() -> Dict[str, Dict[str, Any]]:
        """Return info about available providers"""
        return {
            'embeddinggemma': {
                'type': 'local',
                'dimension': 768,
                'max_tokens': 2048,
                'requires_api_key': False,
            },
            'openai': {
                'type': 'remote',
                'dimension': 768,
                'max_tokens': 8191,
                'requires_api_key': True,
            },
        }
