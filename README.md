# RAG Anywhere

Secure, portable, local-first RAG (Retrieval-Augmented Generation) system with configurable embedding models.

## Features

- 🔒 **Secure**: Fully isolated, can run completely offline
- 🚀 **Fast**: One-command deployment, local vector search with FAISS
- 📦 **Portable**: Multi-platform, minimal dependencies
- 🎯 **Flexible**: Support for multiple embedding models (local and remote)
- 🔧 **Configurable**: Per-database configuration, customizable text splitting

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/rag-anywhere.git
cd rag-anywhere
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package**:
```bash
# Standard installation (CPU-based FAISS)
pip install -e .

# For GPU support (CUDA-enabled FAISS)
pip install -e ".[gpu]"

# For development (includes testing tools)
pip install -e ".[dev]"

# For OpenAI embeddings
pip install -e ".[openai]"

# Install multiple extras
pip install -e ".[gpu,dev,openai]"
```

4. **Verify installation**:
```bash
rag-anywhere --help
```

### Installation Options

RAG Anywhere supports several optional dependency groups that can be installed using the extras syntax:

- **`[gpu]`** - CUDA-enabled FAISS for GPU acceleration
  - Replaces `faiss-cpu` with `faiss-gpu`
  - Requires NVIDIA GPU with CUDA support
  - Significantly faster for large-scale vector operations
  
- **`[openai]`** - OpenAI API integration for embeddings
  - Enables using OpenAI's embedding models
  - Requires API key: `export OPENAI_API_KEY="your-key"`
  
- **`[dev]`** - Development and testing tools
  - pytest, black, ruff, mypy, coverage tools
  - Use when contributing or running tests
  
- **`[api]`** - API server dependencies (already included in base)
  - FastAPI and Uvicorn for REST API
  - Included by default in `requirements.txt`

**Note on GPU Support**: The GPU installation will only provide performance benefits if you have:
- An NVIDIA GPU with CUDA support
- Proper CUDA drivers installed
- Compatible PyTorch with CUDA enabled (installed via sentence-transformers)

If you're unsure, start with the standard CPU installation. You can always reinstall with GPU support later.

## Quick Start Guide

### 1. Create a Database
```bash
# Create a database with local EmbeddingGemma model (default)
rag-anywhere db create my-docs

# Use a local model (offline - will be cached to ~/.rag-anywhere/models/)
rag-anywhere db create my-docs --model ../path/to/local/model

# Use a different HuggingFace model
rag-anywhere db create my-docs --model sentence-transformers/all-MiniLM-L6-v2

# Create with OpenAI embeddings
export OPENAI_API_KEY="your-key"
rag-anywhere db create my-docs --provider openai

# Use OpenAI's larger model
rag-anywhere db create my-docs --provider openai --model text-embedding-3-large
```

### 2. Add Documents
```bash
# Add a single document
rag-anywhere add document.pdf

# Add a directory of documents
rag-anywhere add ./documents/ --recursive

# Add with custom metadata
rag-anywhere add report.pdf --metadata '{"department":"sales","year":2024}'

# Use custom splitter settings
rag-anywhere add document.pdf --splitter recursive --chunk-size 4000
```

### 3. Search
```bash
# Basic search
rag-anywhere search "your query here"

# Advanced search
rag-anywhere search "query" --top-k 10 --min-score 0.7

# Search with context
rag-anywhere search "query" --context 2
```

### 4. Manage Documents
```bash
# List documents
rag-anywhere list

# List with details
rag-anywhere list --verbose

# Remove document
rag-anywhere remove document.pdf
rag-anywhere remove <document-id> --by-id
```

### 5. Database Management
```bash
# List all databases
rag-anywhere db list

# Switch active database
rag-anywhere db use another-db

# View database info
rag-anywhere db info

# Delete database
rag-anywhere db delete old-db
```

## Key Features

### 🔌 Offline-First Design
- **Local Model Support**: Use models completely offline by providing a local path
- **Automatic Caching**: Local models are cached to `~/.rag-anywhere/models/` for stability
- **No Internet Required**: Once models are cached, works entirely offline

### ⚡ Smart Device Detection
- **Automatic GPU/CPU Selection**: Device automatically detected based on installed packages
- **Install-Time Configuration**: Choose CPU or GPU at install time with `pip install -e .[gpu]`
- **No Manual Configuration**: No need to specify device flags - it just works!

### 🗄️ Flexible Database System
- **Multiple Databases**: Create separate databases for different projects
- **Per-Database Models**: Each database can use a different embedding model
- **Easy Switching**: Switch between databases instantly with `rag-anywhere db use`

### 🔍 Hybrid Search
- **Semantic Search**: Dense vector search using embeddings
- **Keyword Search**: Fast FTS5 full-text search with BM25 ranking
- **Boolean Operators**: Support for AND, OR, NOT, phrase queries

## Architecture
```
rag-anywhere/
├── rag_anywhere/
│   ├── core/                 # Core RAG functionality
│   │   ├── embeddings/       # Embedding providers
│   │   ├── splitters/        # Text splitting strategies
│   │   ├── loaders/          # Document loaders
│   │   ├── vector_store.py   # FAISS + SQLite vector storage
│   │   ├── document_store.py # Document management
│   │   ├── indexer.py        # Document indexing pipeline
│   │   └── searcher.py       # Search functionality
│   ├── cli/                  # Command-line interface
│   └── config/               # Configuration management
└── tests/                    # Test suite
```

## Supported File Types

- **Text**: `.txt`, `.md`, `.markdown`
- **PDF**: `.pdf`
- **Word**: `.docx`, `.doc`

More formats coming soon!

## Configuration

RAG Anywhere stores configuration in `~/.rag-anywhere/`:
```
~/.rag-anywhere/
├── config.yaml              # Global config
├── models/                  # Cached local models
│   └── embeddinggemma-300m_abc123/
└── databases/
    ├── my-docs/
    │   ├── rag.db          # SQLite database
    │   └── config.yaml     # Database-specific config
    └── another-db/
        ├── rag.db
        └── config.yaml
```

### Database Configuration

Each database has its own configuration that locks in the embedding model:
```yaml
embedding:
  provider: embeddinggemma
  model: google/embeddinggemma-300m
  dimension: 768
  max_tokens: 2048

splitter:
  defaults:
    .txt:
      strategy: recursive
      chunk_size: 6000
      chunk_overlap: 600
    .pdf:
      strategy: structural
      min_chunk_size: 1000
      max_chunk_size: 6000
```

## Embedding Models

### Local Models (via sentence-transformers)

RAG Anywhere supports any sentence-transformers compatible model. The device (CPU/GPU) is automatically detected based on your installation:

**EmbeddingGemma** (default):
- Model: `google/embeddinggemma-300m`
- Dimensions: 768
- Context: 2048 tokens
- Multi-lingual support
- Size: ~1.2GB
- Usage: `rag-anywhere db create my-db` (default)

**Using Local/Offline Models**:
```bash
# Download model on a different machine, then copy to your system
# RAG Anywhere will cache it to ~/.rag-anywhere/models/
rag-anywhere db create my-db --model /path/to/local/model
```

**Using Different HuggingFace Models**:
```bash
# Any sentence-transformers compatible model
rag-anywhere db create my-db --model sentence-transformers/all-MiniLM-L6-v2
```

**Device Selection**:
- **CPU** (default): Install with `pip install -e .`
- **GPU** (CUDA): Install with `pip install -e .[gpu]`
- Device is auto-detected at runtime based on installed packages
- No manual device configuration needed!

### Remote Models

**OpenAI**:
- Models: `text-embedding-3-small` (default), `text-embedding-3-large`
- Dimensions: 768 (configurable)
- Context: 8191 tokens
- Requires API key: `export OPENAI_API_KEY="your-key"`
- Usage:
  ```bash
  # Small model (default)
  rag-anywhere db create my-db --provider openai

  # Large model
  rag-anywhere db create my-db --provider openai --model text-embedding-3-large
  ```

## License

### Project License
MIT License - see LICENSE file for details

### EmbeddingGemma License
This project uses EmbeddingGemma as the default embedding model, which is provided by Google under the [Gemma Terms of Use](./GEMMA_LICENSE.txt). By using this software, you agree to comply with:

- The [Gemma Terms of Use](./GEMMA_LICENSE.txt)
- The [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)

Users are free to configure alternative embedding models if preferred.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- [x] Core RAG functionality
- [x] CLI interface
- [x] Multiple embedding providers
- [x] Configurable text splitting
- [ ] API server (FastAPI)
- [ ] Hybrid search (dense + sparse)
- [ ] Knowledge graph integration
- [ ] Additional embedding providers
- [ ] Code-aware splitters
- [ ] Web UI
- [ ] Docker support

## Support

For issues and questions, please open an issue on GitHub.
```

## LICENSE
```
# LICENSE
MIT License

Copyright (c) 2025 Andrew O'Shei
