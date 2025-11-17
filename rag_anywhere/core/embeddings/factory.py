# rag_anywhere/core/embeddings/factory.py
from typing import Dict, Any

from .base import EmbeddingProvider
from .providers import (
    EmbeddingGemmaProvider,
    OpenAIEmbeddingProvider,
)
from .utils import DimensionReducerWrapper


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
                model_name=config.get('model', 'google/embeddinggemma-300m'),
                device=config.get('device', None)
            )
        
        elif provider_type == 'openai':
            provider = OpenAIEmbeddingProvider(
                api_key=config['api_key'],
                model=config.get('model', 'text-embedding-3-small')
            )
        
        elif provider_type == 'cohere':
            base_provider = CohereEmbeddingProvider(
                api_key=config['api_key'],
                model=config.get('model', 'embed-english-v3.0')
            )
            # Wrap with dimension reducer
            provider = DimensionReducerWrapper(
                base_provider, 
                target_dim=EmbeddingProviderFactory.STANDARD_DIMENSION
            )
        
        elif provider_type == 'voyage':
            base_provider = VoyageEmbeddingProvider(
                api_key=config['api_key'],
                model=config.get('model', 'voyage-2')
            )
            # Wrap with dimension reducer
            provider = DimensionReducerWrapper(
                base_provider,
                target_dim=EmbeddingProviderFactory.STANDARD_DIMENSION
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
            'cohere': {
                'type': 'remote',
                'dimension': 768,  # After reduction
                'max_tokens': 512,
                'requires_api_key': True,
                'note': 'Dimension reduced from 1024 to 768'
            },
            'voyage': {
                'type': 'remote',
                'dimension': 768,  # After reduction
                'max_tokens': 16000,
                'requires_api_key': True,
                'note': 'Dimension reduced from 1024 to 768'
            },
        }
