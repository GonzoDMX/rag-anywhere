# /python/vector_db_worker.py

import sys
import json
import logging
import os
import numpy as np
import faiss

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("faiss_worker")

# Global State
index = None
current_db_path = None
dimension = 768 # Default for EmbeddingGemma, update via 'init' command

def handle_command(req):
    global index, current_db_path, dimension

    cmd = req.get("command")

    # --- INIT / RESET ---
    if cmd == "init":
        dimension = req.get("dimension", 768)
        # IDMap allows us to store vectors with specific SQLite IDs
        # IndexFlatIP = Inner Product (Cosine Similarity if normalized)
        base_index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(base_index)
        return {"status": "initialized", "count": 0}

    # --- LOAD FROM DISK ---
    elif cmd == "load":
        path = req.get("path")
        if not os.path.exists(path):
            return {"error": "Index file not found", "status": "failed"}
        
        try:
            index = faiss.read_index(path)
            current_db_path = path
            return {"status": "loaded", "count": index.ntotal}
        except Exception as e:
            return {"error": str(e)}

    # --- SAVE TO DISK ---
    elif cmd == "save":
        path = req.get("path", current_db_path)
        if index is None:
            return {"error": "No index in memory"}
        faiss.write_index(index, path)
        return {"status": "saved"}

    # --- ADD VECTORS ---
    elif cmd == "add":
        if index is None: return {"error": "Index not initialized"}
        
        vecs = req.get("vectors") # List of lists
        ids = req.get("ids")      # List of ints
        
        if not vecs or not ids:
            return {"error": "Missing vectors or ids"}

        np_vecs = np.array(vecs, dtype='float32')
        np_ids = np.array(ids, dtype='int64')

        index.add_with_ids(np_vecs, np_ids)
        return {"status": "added", "count": index.ntotal}

    # --- REMOVE VECTORS ---
    elif cmd == "remove":
        if index is None: return {"error": "Index not initialized"}
        
        ids = req.get("ids")
        if not ids: return {"error": "Missing ids"}
        
        np_ids = np.array(ids, dtype='int64')
        index.remove_ids(np_ids)
        return {"status": "removed", "count": index.ntotal}

    # --- SEARCH ---
    elif cmd == "search":
        if index is None: return {"error": "Index not initialized"}
        
        query_vec = req.get("vector")
        k = req.get("top_k", 10)
        
        np_query = np.array([query_vec], dtype='float32')
        
        # D = distances (scores), I = indices (IDs)
        D, I = index.search(np_query, k)
        
        return {
            "results": I[0].tolist(),
            "scores": D[0].tolist()
        }

    return {"error": "Unknown command"}

def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            
            req = json.loads(line)
            resp = handle_command(req)
            
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Faiss Error: {e}")
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()