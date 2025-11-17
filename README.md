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
# Development installation (editable)
pip install -e .

# Or install from requirements
pip install -r requirements.txt
python setup.py develop
```

4. **Verify installation**:
```bash
rag-anywhere --help
```

### Optional Dependencies

**For OpenAI embeddings**:
```bash
pip install openai
export OPENAI_API_KEY="your-api-key"
```

**For API server** (coming soon):
```bash
pip install fastapi uvicorn
```

**For development**:
```bash
pip install -r requirements-dev.txt
```

## Quick Start Guide

### 1. Create a Database
```bash
# Create a database with local EmbeddingGemma model (default)
rag-anywhere db create my-docs

# Or create with OpenAI embeddings
export OPENAI_API_KEY="your-key"
rag-anywhere db create my-docs --provider openai
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

### Local Models

**EmbeddingGemma** (default):
- Model: `google/embeddinggemma-300m`
- Dimensions: 768
- Context: 2048 tokens
- Multi-lingual support
- Size: ~1.2GB

### Remote Models

**OpenAI**:
- Model: `text-embedding-3-small`
- Dimensions: 768 (configurable)
- Context: 8191 tokens
- Requires API key

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
