# examples/client.py
"""
RAG Anywhere - Example Client

This script demonstrates how to interact with the RAG Anywhere API.
It can be used as both a functional test and as reference documentation.

Prerequisites:
    1. Start the server: rag-anywhere server start
    2. Run this script: python examples/client.py

Usage:
    python examples/client.py [--host HOST] [--port PORT]
"""

import requests
from pathlib import Path
from typing import Optional, List, Dict, Any


class RAGAnywhereClient:
    """
    Simple client for the RAG Anywhere API.
    
    Example usage:
        client = RAGAnywhereClient()
        
        # Check server status
        status = client.status()
        print(f"Database: {status['active_database']}")
        
        # Add a document
        result = client.add_document("/path/to/document.pdf")
        doc_id = result['document_id']
        
        # Search
        results = client.search("your query here")
        for r in results:
            print(f"{r['document']['filename']}: {r['content'][:100]}...")
        
        # Remove document
        client.remove_document(doc_id)
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize the client.
        
        Args:
            host: Server host address
            port: Server port
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = 30
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        response = requests.request(method, url, **kwargs)
        
        if response.status_code >= 400:
            error_detail = response.json().get('detail', response.text)
            raise Exception(f"API Error ({response.status_code}): {error_detail}")
        
        return response.json()
    
    # -------------------------------------------------------------------------
    # Status & Admin
    # -------------------------------------------------------------------------
    
    def status(self) -> dict:
        """
        Get server status.
        
        Returns:
            dict with keys: status, active_database, num_documents, num_vectors
        
        Example:
            >>> client.status()
            {
                'status': 'running',
                'active_database': 'my-docs',
                'num_documents': 10,
                'num_vectors': 150
            }
        """
        return self._request("GET", "/admin/status")
    
    def is_healthy(self) -> bool:
        """
        Check if server is healthy.
        
        Returns:
            True if server is running and responsive
        """
        try:
            status = self.status()
            return status.get('status') == 'running'
        except Exception:
            return False
    
    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------
    
    def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        splitter_overrides: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        Add a document to the database.
        
        Args:
            file_path: Absolute path to the document file
            metadata: Optional metadata to attach to the document
            splitter_overrides: Optional splitter configuration
        
        Returns:
            dict with keys: status, document_id, filename
        
        Example:
            >>> client.add_document(
            ...     "/home/user/docs/report.pdf",
            ...     metadata={"department": "sales", "year": 2024}
            ... )
            {
                'status': 'success',
                'document_id': 'abc-123-...',
                'filename': 'report.pdf'
            }
        """
        payload = {
            'file_path': str(file_path),
            'metadata': metadata,
            'splitter_overrides': splitter_overrides
        }
        return self._request("POST", "/documents/add", json=payload)
    
    def remove_document(self, document_id: str) -> dict:
        """
        Remove a document from the database.
        
        Args:
            document_id: UUID of the document to remove
        
        Returns:
            dict with key: status
        
        Example:
            >>> client.remove_document("abc-123-...")
            {'status': 'success'}
        """
        payload = {'document_id': document_id}
        return self._request("POST", "/documents/remove", json=payload)
    
    def list_documents(self) -> List[dict]:
        """
        List all documents in the database.
        
        Returns:
            List of document dicts with keys: id, filename, created_at, metadata, num_chunks
        
        Example:
            >>> docs = client.list_documents()
            >>> for doc in docs:
            ...     print(f"{doc['filename']} ({doc['num_chunks']} chunks)")
        """
        response = self._request("GET", "/documents/list")
        return response.get('documents', [])
    
    def get_document(self, document_id: str) -> dict:
        """
        Get detailed information about a document.
        
        Args:
            document_id: UUID of the document
        
        Returns:
            Document dict with full content and chunks
        """
        return self._request("GET", f"/documents/{document_id}")
    
    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default: 5)
            min_score: Minimum similarity score threshold (0.0-1.0)
        
        Returns:
            List of search result dicts with keys:
                - chunk_id
                - content
                - similarity_score
                - document (id, filename)
                - position (chunk_index, start_char, end_char)
                - metadata
        
        Example:
            >>> results = client.search("quarterly revenue", top_k=3)
            >>> for r in results:
            ...     print(f"[{r['similarity_score']:.2f}] {r['document']['filename']}")
            ...     print(f"  {r['content'][:200]}...")
        """
        payload = {
            'query': query,
            'top_k': top_k,
            'min_score': min_score
        }
        response = self._request("POST", "/search", json=payload)
        return response.get('results', [])


# =============================================================================
# Example Usage & Functional Test
# =============================================================================

def run_example():
    """
    Run example usage demonstrating all API features.
    
    This serves as both documentation and a functional test.
    """
    print("=" * 60)
    print("RAG Anywhere - Example Client")
    print("=" * 60)
    
    # Initialize client
    client = RAGAnywhereClient()
    
    # -------------------------------------------------------------------------
    # 1. Check Server Status
    # -------------------------------------------------------------------------
    print("\n[1] Checking server status...")
    
    if not client.is_healthy():
        print("✗ Server is not running!")
        print("  Start it with: rag-anywhere server start")
        return False
    
    status = client.status()
    print(f"✓ Server is running")
    print(f"  Database: {status['active_database']}")
    print(f"  Documents: {status['num_documents']}")
    print(f"  Vectors: {status['num_vectors']}")
    
    # -------------------------------------------------------------------------
    # 2. List Existing Documents
    # -------------------------------------------------------------------------
    print("\n[2] Listing documents...")
    
    documents = client.list_documents()
    
    if documents:
        print(f"✓ Found {len(documents)} document(s):")
        for doc in documents[:5]:  # Show first 5
            print(f"  - {doc['filename']} ({doc['num_chunks']} chunks)")
        if len(documents) > 5:
            print(f"  ... and {len(documents) - 5} more")
    else:
        print("  No documents in database")
    
    # -------------------------------------------------------------------------
    # 3. Add a Test Document (if exists)
    # -------------------------------------------------------------------------
    print("\n[3] Adding a test document...")
    
    # Look for a test file
    test_files = [
        Path("./test_document.txt"),
        Path("../README.md"),
        Path("../LICENSE"),
    ]
    
    test_file = None
    for f in test_files:
        if f.exists():
            test_file = f
            break
    
    added_doc_id = None
    if test_file:
        try:
            result = client.add_document(
                str(test_file.absolute()),
                metadata={"type": "test", "source": "example_client"}
            )
            added_doc_id = result['document_id']
            print(f"✓ Added: {result['filename']}")
            print(f"  Document ID: {added_doc_id}")
        except Exception as e:
            print(f"⚠ Could not add document: {e}")
    else:
        print("  No test file found, skipping...")
    
    # -------------------------------------------------------------------------
    # 4. Search
    # -------------------------------------------------------------------------
    print("\n[4] Performing search...")
    
    # Use a generic query that might match most documents
    query = "gemma usage terms"
    results = client.search(query, top_k=3)
    
    if results:
        print(f"✓ Found {len(results)} result(s) for '{query}':")
        for i, r in enumerate(results, 1):
            score = r['similarity_score']
            filename = r['document']['filename']
            content_preview = r['content'][:100].replace('\n', ' ')
            print(f"\n  Result {i} (score: {score:.3f})")
            print(f"  File: {filename}")
            print(f"  Content: {content_preview}...")
    else:
        print(f"  No results for '{query}'")
    
    # -------------------------------------------------------------------------
    # 5. Clean Up (Remove Test Document)
    # -------------------------------------------------------------------------
    if added_doc_id:
        print("\n[5] Cleaning up test document...")
        
        try:
            client.remove_document(added_doc_id)
            print(f"✓ Removed test document: {added_doc_id}")
        except Exception as e:
            print(f"⚠ Could not remove document: {e}")
    
    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)
    
    return True


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG Anywhere Example Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python examples/client.py
    
    # Connect to different port
    python examples/client.py --port 9000
    
    # Connect to remote server
    python examples/client.py --host 192.168.1.100 --port 8000
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Server host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Server port (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Update client defaults
    RAGAnywhereClient.__init__.__defaults__ = (args.host, args.port)
    
    # Run example
    success = run_example()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
