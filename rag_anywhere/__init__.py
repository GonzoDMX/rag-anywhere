# rag_anywhere/__init__.py

import os
import platform

# Fix for OpenMP runtime conflict on macOS
# Multiple libraries (PyTorch, FAISS, scikit-learn) bundle their own libomp.dylib
# which causes thread synchronization crashes when they try to manage threads together.
# These settings must be applied before importing torch, faiss, or any modules that depend on them.
if platform.system() == 'Darwin':  # macOS only
    # Suppress OpenMP library duplication warning
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Force single-threaded OpenMP execution to prevent thread synchronization crashes
    # This prevents segfaults in __kmp_suspend_64 and other OpenMP thread barriers
    os.environ['OMP_NUM_THREADS'] = '1'

