# rag_anywhere/cli/context.py
from pathlib import Path
from typing import Optional
import os

from ..config import Config
from ..core import (
    EmbeddingProviderFactory,
    LoaderRegistry,
    DocumentStore,
    VectorStore,
    Indexer,
    Searcher
)


class RAGContext:
    """
    Manages the active database and loaded resources.
    Handles model loading/unloading when switching databases.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config = Config(config_dir)
        
        # Currently loaded resources
        self.active_db_name: Optional[str] = None
        self.db_config: Optional[dict] = None
        self.embedding_provider = None
        self.loader_registry = None
        self.document_store = None
        self.vector_store = None
        self.indexer = None
        self.searcher = None
        
        # Track loaded embedding model for reuse
        self._loaded_embedding_model = None
    
    def get_active_database_name(self) -> Optional[str]:
        """Get the name of the currently active database"""
        if self.active_db_name:
            return self.active_db_name
        return self.config.get_active_database()
    
    def ensure_active_database(self):
        """Ensure there is an active database, raise error if not"""
        if not self.get_active_database_name():
            raise ValueError(
                "No active database. Create one with 'rag-anywhere db create <name>' "
                "or activate one with 'rag-anywhere db use <name>'"
            )
    
    def load_database(self, db_name: str, verbose: bool = True):
        """
        Load a database and its resources.
        Handles embedding model loading/reloading.
        """
        if not self.config.database_exists(db_name):
            raise ValueError(f"Database '{db_name}' does not exist")
        
        # Load database config
        self.db_config = self.config.load_database_config(db_name)
        
        # Check if we need to reload embedding model
        current_model_key = self._get_model_key()
        new_model_key = self._get_model_key_from_config(self.db_config)
        
        if current_model_key != new_model_key:
            if verbose:
                if current_model_key:
                    print(f"Switching embedding model from {current_model_key} to {new_model_key}")
                else:
                    print(f"Loading embedding model: {new_model_key}")
            
            # Unload old model
            self.embedding_provider = None
            self._loaded_embedding_model = None
            
            # Load new model
            self.embedding_provider = self._create_embedding_provider()
            self._loaded_embedding_model = new_model_key
        elif verbose and self.embedding_provider:
            print(f"Embedding model already loaded: {current_model_key}")
        elif not self.embedding_provider:
            if verbose:
                print(f"Loading embedding model: {new_model_key}")
            self.embedding_provider = self._create_embedding_provider()
            self._loaded_embedding_model = new_model_key
        
        # Initialize loader registry
        self.loader_registry = LoaderRegistry()
        
        # Initialize document and vector stores
        db_path = str(self.config.get_database_db_path(db_name))
        self.document_store = DocumentStore(db_path)
        self.vector_store = VectorStore(
            db_path,
            dimension=self.db_config['embedding']['dimension']
        )
        
        # Initialize indexer and searcher
        self.indexer = Indexer(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            loader_registry=self.loader_registry
        )
        
        self.searcher = Searcher(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )
        
        # Set as active
        self.active_db_name = db_name
        self.config.set_active_database(db_name)
        
        if verbose:
            print(f"✓ Database '{db_name}' is now active")
    
    def _create_embedding_provider(self):
        """Create embedding provider from current config"""
        embedding_config = self.db_config['embedding'].copy()
        
        # Add API key from environment if needed
        provider = embedding_config['provider']
        if provider == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenAI provider requires OPENAI_API_KEY environment variable"
                )
            embedding_config['api_key'] = api_key
        
        return EmbeddingProviderFactory.create_provider(embedding_config)
    
    def _get_model_key(self) -> Optional[str]:
        """Get identifier for currently loaded model"""
        return self._loaded_embedding_model
    
    def _get_model_key_from_config(self, config: dict) -> str:
        """Get model identifier from config"""
        emb = config['embedding']
        return f"{emb['provider']}:{emb['model']}"
    
    def get_splitter_config(self, file_extension: str, overrides: Optional[dict] = None) -> dict:
        """
        Get splitter configuration for a file type with optional overrides
        
        Args:
            file_extension: File extension (e.g., '.pdf')
            overrides: Optional parameter overrides from CLI/API
        
        Returns:
            Splitter configuration dict
        """
        self.ensure_active_database()
        
        # Get defaults for file type
        config = self.config.get_splitter_config_for_file(
            self.active_db_name,
            file_extension
        )
        
        # Apply overrides
        if overrides:
            config.update(overrides)
        
        return config
