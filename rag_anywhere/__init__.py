# rag_anywhere/__init__.py

import os

# Fix for OpenMP runtime conflict on macOS
# This prevents crashes when both PyTorch and FAISS load their own copies of libomp
# Must be set before importing torch, faiss, or any modules that depend on them
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

