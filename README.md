# RAG Anywhere

Secure, portable, local-first RAG (Retrieval-Augmented Generation) system powered by EmbeddingGemma.

## Features

- 🔒 **Secure**: Fully isolated, runs completely offline with local embeddings
- 🚀 **Fast**: One-command deployment, local vector search with FAISS
- 📦 **Portable**: Multi-platform, minimal dependencies, no API keys required
- 🎯 **Task-Optimized**: Specialized search modes for different use cases (code, Q&A, facts)
- 🔧 **Configurable**: Customizable text splitting, document types, metadata

## Installation

### Prerequisites

- Python 3.12 or higher
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

# Install multiple extras
pip install -e ".[gpu,dev]"
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

#### Semantic Search (Default)
```bash
# Basic search
rag-anywhere search "your query here"

# Advanced search
rag-anywhere search "query" --top-k 10 --min-score 0.7

# Search with context
rag-anywhere search "query" --context 2
```

#### Keyword Search
RAG Anywhere provides powerful keyword search using SQLite FTS5 with two modes:

**Free-form Mode** (supports FTS5 query syntax):
```bash
# Simple query
rag-anywhere search keyword "machine learning"

# Boolean AND
rag-anywhere search keyword "machine AND learning"

# Boolean OR
rag-anywhere search keyword "machine OR learning"

# NOT operator
rag-anywhere search keyword "machine NOT cat"

# Phrase queries (exact match)
rag-anywhere search keyword '"machine learning"'

# Prefix matching
rag-anywhere search keyword "mach*"

# Exact match flag
rag-anywhere search keyword "Google's" --exact-match

# Exclude terms
rag-anywhere search keyword "dog" --exclude "cat,bird"

# More results
rag-anywhere search keyword "machine" --top-k 20

# Disable highlighting
rag-anywhere search keyword "machine" --no-highlight
```

**Structured Mode** (explicit required/optional/exclude):
```bash
# Required keywords (all must be present - AND logic)
rag-anywhere search keyword --required "machine,learning"

# Optional keywords (at least one must be present - OR logic)
rag-anywhere search keyword --optional "machine,learning,neural"

# Combine required and optional
rag-anywhere search keyword --required "learning" --optional "machine,deep,neural"

# Exclude keywords
rag-anywhere search keyword --required "dog" --exclude "cat,bird"

# Full example
rag-anywhere search keyword \
  --required "machine,learning" \
  --optional "neural,deep" \
  --exclude "statistics" \
  --top-k 15
```

**Tips:**
- Free-form mode gives you more control with FTS5 syntax
- Structured mode is simpler and less error-prone
- Use `--highlight` (default) to see matched terms with `<mark>` tags
- Keyword search is faster than semantic search for exact term matching
- Combine with `--metadata` to see chunk metadata

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

## API Server

RAG Anywhere includes a FastAPI-based REST API server for programmatic access.

### Starting the Server

```bash
# Start server with default settings (port 8000)
rag-anywhere server --db my-docs

# Custom port
rag-anywhere server --db my-docs --port 8080

# Server management
rag-anywhere server status   # Check server status
rag-anywhere server stop     # Stop the server
rag-anywhere server restart  # Restart the server
```

### API Endpoints

#### Keyword Search (Unified)
**Endpoint**: `POST /search/keyword`

Supports both free-form and structured modes via automatic mode detection.

**Free-form mode example**:
```bash
curl -X POST "http://localhost:8000/search/keyword" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine AND learning",
    "top_k": 10,
    "highlight": true,
    "exact_match": false,
    "exclude_terms": ["statistics"]
  }'
```

**Structured mode example**:
```bash
curl -X POST "http://localhost:8000/search/keyword" \
  -H "Content-Type: application/json" \
  -d '{
    "required_keywords": ["machine", "learning"],
    "optional_keywords": ["neural", "deep"],
    "exclude_keywords": ["statistics"],
    "top_k": 10,
    "highlight": true
  }'
```

**Request Parameters**:

*Free-form mode*:
- `query` (string, required): FTS5 query syntax
- `exclude_terms` (array, optional): Terms to exclude
- `exact_match` (boolean, optional): Treat as exact phrase
- `top_k` (integer, optional): Number of results (default: 10)
- `highlight` (boolean, optional): Highlight matches (default: false)

*Structured mode*:
- `required_keywords` (array, optional): All must be present (AND)
- `optional_keywords` (array, optional): At least one present (OR)
- `exclude_keywords` (array, optional): None present (NOT)
- `top_k` (integer, optional): Number of results (default: 10)
- `highlight` (boolean, optional): Highlight matches (default: false)

**Response**:
```json
{
  "results": [
    {
      "chunk_id": "doc-uuid_0",
      "content": "Machine learning is...",
      "score": 2.5,
      "document": {
        "id": "doc-uuid",
        "filename": "ml-intro.txt"
      },
      "position": {
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 500
      },
      "metadata": {}
    }
  ],
  "query": "machine AND learning"
}
```

#### Semantic Search
**Endpoint**: `POST /search`

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks and deep learning",
    "top_k": 5,
    "min_score": 0.7
  }'
```

#### Document Management

**Add Document**: `POST /documents/add`
```bash
curl -X POST "http://localhost:8000/documents/add" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "metadata": {"category": "ml"}
  }'
```

**List Documents**: `GET /documents/list`
```bash
curl "http://localhost:8000/documents/list"
```

**Remove Document**: `POST /documents/remove`
```bash
curl -X POST "http://localhost:8000/documents/remove" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc-uuid"}'
```

**Batch Add**: `POST /documents/add-batch`
```bash
curl -X POST "http://localhost:8000/documents/add-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"file_path": "/path/to/doc1.pdf"},
      {"file_path": "/path/to/doc2.txt"}
    ],
    "fail_fast": false
  }'
```

For complete API documentation, start the server and visit `http://localhost:8000/docs` for the interactive Swagger UI.

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
- [x] API server (FastAPI)
- [x] Keyword search (FTS5 with BM25 ranking)
- [x] Hybrid search capabilities (semantic + keyword)
- [x] Knowledge graph integration
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
