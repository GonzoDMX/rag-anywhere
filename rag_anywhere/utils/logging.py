# rag_anywhere/utils/logging.py
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(config_dir: Path, debug: bool = False):
    """Setup logging for RAG Anywhere"""
    log_dir = config_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"rag-anywhere.log"
    
    # Configure logging
    level = logging.DEBUG if debug else logging.INFO
    
    # File handler - always detailed
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - less verbose unless debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger('rag_anywhere')
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(f'rag_anywhere.{name}')
