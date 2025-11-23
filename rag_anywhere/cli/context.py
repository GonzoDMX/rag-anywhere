# rag_anywhere/cli/context.py

import os
from pathlib import Path
from typing import Optional

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
from ..core.entity_store import EntityStore
from ..core.gliner import GLiNERExtractor, GLiNERSubChunker, GLiNERBatchProcessor
from ..utils.logging import get_logger

logger = get_logger('cli.context')


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
        self.entity_store = None
        self.gliner_extractor = None
        self.gliner_processor = None
        self.indexer = None
        self.searcher = None

        # Track loaded models for reuse
        self._loaded_embedding_model = None
        self._loaded_gliner_model = None
    
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

    @property
    def safe_entity_store(self) -> EntityStore:
        """Get entity store with type assertion for type checker"""
        if self.entity_store is None:
            raise RuntimeError("Entity store not initialized. Load a database first.")
        return self.entity_store

    @property
    def safe_gliner_processor(self) -> GLiNERBatchProcessor:
        """Get GLiNER processor with type assertion for type checker"""
        if self.gliner_processor is None:
            raise RuntimeError("GLiNER processor not initialized. Load a database first.")
        return self.gliner_processor
    
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

    def load_entity_store_only(self, db_name: str) -> EntityStore:
        """
        Load only the EntityStore without loading embedding models or other heavy resources.
        Useful for KG operations that don't need embeddings.

        Args:
            db_name: Database name

        Returns:
            EntityStore instance
        """
        logger.debug(f"Loading entity store only for database '{db_name}'")

        if not self.config.database_exists(db_name):
            raise ValueError(f"Database '{db_name}' does not exist")

        # Load database config to check if GLiNER is enabled
        db_config = self.config.load_database_config(db_name)
        gliner_config = db_config.get('gliner', {})
        gliner_enabled = gliner_config.get('enabled', True)

        if not gliner_enabled:
            raise ValueError(f"Knowledge graph is disabled for database '{db_name}'")

        # Get database path and create EntityStore
        db_path = str(self.config.get_database_db_path(db_name))
        return EntityStore(db_path)
    
    def load_database(self, db_name: str, verbose: bool = True):
        """
        Load a database and its resources.
        Handles embedding model loading/reloading.
        """
        logger.info(f"Loading database '{db_name}'")

        if not self.config.database_exists(db_name):
            logger.error(f"Database '{db_name}' does not exist")
            raise ValueError(f"Database '{db_name}' does not exist")

        try:
            # Load database config
            logger.debug(f"Loading database configuration for '{db_name}'")
            self.db_config = self.config.load_database_config(db_name)
            logger.debug(f"Database config loaded: provider={self.db_config['embedding']['provider']}, model={self.db_config['embedding']['model']}")

            # Check if we need to reload embedding model
            current_model_key = self._get_model_key()
            new_model_key = self._get_model_key_from_config(self.db_config)
            logger.debug(f"Current model: {current_model_key}, New model: {new_model_key}")

            if current_model_key != new_model_key:
                if verbose:
                    if current_model_key:
                        print(f"Switching embedding model from {current_model_key} to {new_model_key}")
                        logger.info(f"Switching embedding model from {current_model_key} to {new_model_key}")
                    else:
                        print(f"Loading embedding model: {new_model_key}")
                        logger.info(f"Loading embedding model: {new_model_key}")

                # Unload old model
                logger.debug("Unloading old embedding model")
                self.embedding_provider = None
                self._loaded_embedding_model = None

                # Load new model
                logger.debug("Creating new embedding provider")
                self.embedding_provider = self._create_embedding_provider()
                self._loaded_embedding_model = new_model_key
                logger.info("Embedding provider created successfully")

            elif verbose and self.embedding_provider:
                print(f"Embedding model already loaded: {current_model_key}")
                logger.debug(f"Embedding model already loaded: {current_model_key}")
            elif not self.embedding_provider:
                if verbose:
                    print(f"Loading embedding model: {new_model_key}")
                logger.info(f"Loading embedding model: {new_model_key}")
                self.embedding_provider = self._create_embedding_provider()
                self._loaded_embedding_model = new_model_key
                logger.info("Embedding provider created successfully")

            # Initialize loader registry
            logger.debug("Initializing loader registry")
            self.loader_registry = LoaderRegistry()

            # Initialize document and vector stores
            db_path = str(self.config.get_database_db_path(db_name))
            logger.debug(f"Initializing document store at {db_path}")
            self.document_store = DocumentStore(db_path)

            logger.debug(f"Initializing vector store with dimension={self.db_config['embedding']['dimension']}")
            self.vector_store = VectorStore(
                db_path,
                dimension=self.db_config['embedding']['dimension']
            )

            # Initialize keyword searcher
            logger.debug("Initializing keyword searcher")
            self.keyword_searcher = KeywordSearcher(db_path)

            # Initialize entity store and GLiNER (if enabled)
            gliner_config = self.db_config.get('gliner', {})
            gliner_enabled = gliner_config.get('enabled', True)

            if gliner_enabled:
                logger.debug("Initializing entity store")
                self.entity_store = EntityStore(db_path)

                # Check if we need to reload GLiNER model
                new_gliner_key = gliner_config.get('model_size', 'multi')
                if self._loaded_gliner_model != new_gliner_key:
                    if verbose and self._loaded_gliner_model:
                        print(f"Switching GLiNER model from {self._loaded_gliner_model} to {new_gliner_key}")
                    elif verbose:
                        print(f"Loading GLiNER model: {new_gliner_key}")

                    logger.debug(f"Loading GLiNER model: {new_gliner_key}")
                    self.gliner_extractor = GLiNERExtractor(
                        model_size=new_gliner_key,
                        confidence_threshold=gliner_config.get('confidence_threshold', 0.5),
                        device='cpu',  # TODO: detect GPU availability
                        cache_dir=str(self.config.gliner_models_dir)
                    )

                    sub_chunker = GLiNERSubChunker(
                        word_size=gliner_config.get('subchunk_word_size', 320),
                        overlap=gliner_config.get('subchunk_overlap', 10)
                    )

                    self.gliner_processor = GLiNERBatchProcessor(
                        extractor=self.gliner_extractor,
                        sub_chunker=sub_chunker,
                        max_labels_per_pass=gliner_config.get('max_labels_per_pass', 10)
                    )

                    self._loaded_gliner_model = new_gliner_key
                    logger.info(f"GLiNER model loaded: {new_gliner_key}")
                elif verbose and self.gliner_extractor:
                    print(f"GLiNER model already loaded: {new_gliner_key}")
                    logger.debug(f"GLiNER model already loaded: {new_gliner_key}")
            else:
                logger.debug("GLiNER disabled for this database")
                self.entity_store = None
                self.gliner_extractor = None
                self.gliner_processor = None
                self._loaded_gliner_model = None

            # Initialize indexer and searcher
            logger.debug("Initializing indexer")
            self.indexer = Indexer(
                document_store=self.document_store,
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                keyword_searcher=self.keyword_searcher,
                entity_store=self.entity_store,
                gliner_processor=self.gliner_processor,
                gliner_config=gliner_config,
                loader_registry=self.loader_registry
            )

            logger.debug("Initializing searcher")
            self.searcher = Searcher(
                document_store=self.document_store,
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider
            )

            # Set as active
            self.active_db_name = db_name
            self.config.set_active_database(db_name)
            logger.info(f"Database '{db_name}' loaded and set as active")

            if verbose:
                print(f"✓ Database '{db_name}' is now active")

        except Exception as e:
            logger.error(f"Failed to load database '{db_name}': {type(e).__name__}: {e}", exc_info=True)
            raise
    
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
