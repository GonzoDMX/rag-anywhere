# rag_anywhere/config/settings.py

import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Config:
    """Configuration management for RAG Anywhere"""
    
    DEFAULT_CONFIG_DIR = Path.home() / ".rag-anywhere"
    DEFAULT_DATABASES_DIR = DEFAULT_CONFIG_DIR / "databases"
    
    # Default splitter configurations per file type
    DEFAULT_SPLITTER_CONFIG = {
        '.txt': {
            'strategy': 'recursive',
            'chunk_size': 6000,
            'chunk_overlap': 600
        },
        '.md': {
            'strategy': 'recursive',
            'chunk_size': 6000,
            'chunk_overlap': 600
        },
        '.markdown': {
            'strategy': 'recursive',
            'chunk_size': 6000,
            'chunk_overlap': 600
        },
        '.pdf': {
            'strategy': 'structural',
            'min_chunk_size': 1000,
            'max_chunk_size': 6000
        },
        '.docx': {
            'strategy': 'structural',
            'min_chunk_size': 1000,
            'max_chunk_size': 6000
        },
        '.doc': {
            'strategy': 'structural',
            'min_chunk_size': 1000,
            'max_chunk_size': 6000
        }
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.databases_dir = self.config_dir / "databases"
        self.global_config_path = self.config_dir / "config.yaml"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure configuration directories exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.databases_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file"""
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _save_yaml(self, path: Path, data: Dict[str, Any]):
        """Save YAML file"""
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    # Global config methods
    
    def load_global_config(self) -> Dict[str, Any]:
        """Load global configuration"""
        return self._load_yaml(self.global_config_path)
    
    def save_global_config(self, config: Dict[str, Any]):
        """Save global configuration"""
        self._save_yaml(self.global_config_path, config)
    
    def get_active_database(self) -> Optional[str]:
        """Get name of active database"""
        config = self.load_global_config()
        return config.get('active_database')
    
    def set_active_database(self, db_name: str):
        """Set active database"""
        config = self.load_global_config()
        config['active_database'] = db_name
        self.save_global_config(config)
    
    # Database-specific config methods
    
    def get_database_dir(self, db_name: str) -> Path:
        """Get directory for a specific database"""
        return self.databases_dir / db_name
    
    def get_database_config_path(self, db_name: str) -> Path:
        """Get config file path for a database"""
        return self.get_database_dir(db_name) / "config.yaml"
    
    def get_database_db_path(self, db_name: str) -> Path:
        """Get SQLite database path for a database"""
        return self.get_database_dir(db_name) / "rag.db"
    
    def database_exists(self, db_name: str) -> bool:
        """Check if database exists"""
        return self.get_database_dir(db_name).exists()
    
    def list_databases(self) -> list[str]:
        """List all database names"""
        if not self.databases_dir.exists():
            return []
        return [
            d.name for d in self.databases_dir.iterdir() 
            if d.is_dir() and (d / "config.yaml").exists()
        ]
    
    def create_database_config(
        self,
        db_name: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_max_tokens: int,
        additional_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new database configuration"""
        db_dir = self.get_database_dir(db_name)
        db_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            'database': {
                'name': db_name,
                'created_at': datetime.now().isoformat(),
                'version': '0.1.0'
            },
            'embedding': {
                'provider': embedding_provider,
                'model': embedding_model,
                'dimension': embedding_dimension,
                'max_tokens': embedding_max_tokens
            },
            'splitter': {
                'defaults': self.DEFAULT_SPLITTER_CONFIG.copy()
            },
            'vector_store': {
                'metric': 'cosine'
            }
        }
        
        # Merge additional config
        if additional_config:
            for key, value in additional_config.items():
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
        
        self._save_yaml(self.get_database_config_path(db_name), config)
        return config
    
    def load_database_config(self, db_name: str) -> Dict[str, Any]:
        """Load database configuration"""
        config_path = self.get_database_config_path(db_name)
        if not config_path.exists():
            raise ValueError(f"Database '{db_name}' does not exist")
        return self._load_yaml(config_path)
    
    def save_database_config(self, db_name: str, config: Dict[str, Any]):
        """Save database configuration"""
        self._save_yaml(self.get_database_config_path(db_name), config)
    
    def delete_database(self, db_name: str):
        """Delete a database and all its files"""
        db_dir = self.get_database_dir(db_name)
        if db_dir.exists():
            shutil.rmtree(db_dir)
        
        # If this was the active database, clear it
        if self.get_active_database() == db_name:
            config = self.load_global_config()
            config.pop('active_database', None)
            self.save_global_config(config)
    
    def get_splitter_config_for_file(
        self, 
        db_name: str, 
        file_extension: str
    ) -> Dict[str, Any]:
        """Get splitter configuration for a specific file type"""
        config = self.load_database_config(db_name)
        defaults = config.get('splitter', {}).get('defaults', {})
        
        # Return file-type specific config or fall back to default recursive
        if file_extension in defaults:
            return defaults[file_extension].copy()
        else:
            # Fallback to recursive splitter
            return {
                'strategy': 'recursive',
                'chunk_size': 6000,
                'chunk_overlap': 600
            }
