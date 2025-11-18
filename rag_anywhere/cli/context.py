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
from ..core.keyword_search import KeywordSearcher


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
        self.keyword_searcher = None
        self.indexer = None
        self.searcher = None

        # Track loaded embedding model for reuse
        self._loaded_embedding_model = None
    
    @property
    def safe_indexer(self) -> Indexer:
        """Get indexer with type assertion for type checker"""
        if self.indexer is None:
            raise RuntimeError("Indexer not initialized. Load a database first.")
        return self.indexer
    
    @property
    def safe_searcher(self) -> Searcher:
        """Get searcher with type assertion for type checker"""
        if self.searcher is None:
            raise RuntimeError("Searcher not initialized. Load a database first.")
        return self.searcher
    
    @property
    def safe_document_store(self) -> DocumentStore:
        """Get document store with type assertion for type checker"""
        if self.document_store is None:
            raise RuntimeError("Document store not initialized. Load a database first.")
        return self.document_store
    
    @property
    def safe_keyword_searcher(self) -> KeywordSearcher:
        """Get keyword searcher with type assertion for type checker"""
        if self.keyword_searcher is None:
            raise RuntimeError("Keyword searcher not initialized. Load a database first.")
        return self.keyword_searcher
    
    @property
    def safe_embedding_provider(self):
        """Get embedding provider with type assertion for type checker"""
        if self.embedding_provider is None:
            raise RuntimeError("Embedding provider not initialized. Load a database first.")
        return self.embedding_provider
    
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

        # Initialize keyword searcher
        self.keyword_searcher = KeywordSearcher(db_path)

        # Initialize indexer and searcher
        self.indexer = Indexer(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            keyword_searcher=self.keyword_searcher,
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
        if self.db_config is None:
            raise ValueError("Database config not loaded")
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
            self.active_db_name,  # type: ignore[arg-type]
            file_extension
        )
        # Ensure active_db_name is not None
        if self.active_db_name is None:
            raise ValueError("No active database")
        
        # Get defaults for file type
        config = self.config.get_splitter_config_for_file(
            self.active_db_name,
            file_extension
        )        
        # Apply overrides
        if overrides:
            config.update(overrides)
        
        return config
