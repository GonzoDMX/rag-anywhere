#!/usr/bin/env python3
"""
API Testing Script for RAG Anywhere

Tests all major API endpoints including:
- Document management (add, list, remove)
- Batch processing
- Semantic search
- Keyword search (free-form and structured modes)
"""

import sys
import argparse
import requests
from pathlib import Path
import tempfile


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_test(name):
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {name}")


def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")


class APITester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.test_files = []
        self.document_ids = []

    def setup_test_files(self):
        """Create temporary test files"""
        print_test("Setting up test files")

        temp_dir = Path(tempfile.mkdtemp(prefix="rag_test_"))
        self.test_dir = temp_dir

        # Create test files
        files = {
            "machine_learning.txt": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "deep_learning.txt": "Deep learning uses neural networks with multiple layers to analyze data patterns.",
            "nlp.txt": "Natural language processing helps computers understand and process human language.",
            "google_test.txt": "Google's search engine uses advanced algorithms for indexing and retrieval.",
        }

        for filename, content in files.items():
            file_path = temp_dir / filename
            file_path.write_text(content)
            self.test_files.append(file_path)

        print_success(f"Created {len(self.test_files)} test files in {temp_dir}")

    def test_single_document_add(self):
        """Test adding a single document"""
        print_test("Single document add")

        file_path = str(self.test_files[0].absolute())
        response = requests.post(
            f"{self.base_url}/documents/add",
            json={
                "file_path": file_path,
                "metadata": {"category": "ml", "test": True}
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            self.document_ids.append(data["document_id"])
            print_success(f"Added document: {data['filename']} (ID: {data['document_id'][:8]}...)")
            return True
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def test_batch_document_add(self):
        """Test batch document processing"""
        print_test("Batch document add")

        documents = [
            {"file_path": str(f.absolute()), "metadata": {"test": True}}
            for f in self.test_files[1:]  # Skip first file (already added)
        ]

        response = requests.post(
            f"{self.base_url}/documents/add-batch",
            json={
                "documents": documents,
                "fail_fast": False
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            summary = data["summary"]
            print_success(f"Batch add: {summary['succeeded']} succeeded, {summary['failed']} failed")

            for result in data["results"]:
                if result["status"] == "success":
                    self.document_ids.append(result["document_id"])

            return summary['failed'] == 0
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def test_list_documents(self):
        """Test listing documents"""
        print_test("List documents")

        response = requests.get(f"{self.base_url}/documents/list", timeout=10)

        if response.status_code == 200:
            data = response.json()
            doc_count = len(data["documents"])
            print_success(f"Listed {doc_count} documents")
            return doc_count > 0
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def test_semantic_search(self):
        """Test semantic search"""
        print_test("Semantic search")

        response = requests.post(
            f"{self.base_url}/search",
            json={
                "query": "neural networks and deep learning",
                "top_k": 5
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            result_count = len(data["results"])
            print_success(f"Found {result_count} results")

            if result_count > 0:
                top_result = data["results"][0]
                print(f"  Top result: {top_result['document']['filename']}")
                print(f"  Score: {top_result['similarity_score']:.3f}")

            return result_count > 0
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def test_keyword_search(self):
        """Test keyword search"""
        print_test("Keyword search")

        response = requests.post(
            f"{self.base_url}/search/keyword",
            json={
                "query": "machine learning",
                "top_k": 5,
                "highlight": True
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            result_count = len(data["results"])
            print_success(f"Found {result_count} results")

            if result_count > 0:
                top_result = data["results"][0]
                print(f"  Top result: {top_result['document']['filename']}")
                print(f"  Score: {top_result['score']:.3f}")
                # Check if highlighting is present
                if "<mark>" in top_result['content']:
                    print_success("  Highlighting is working")

            return result_count > 0
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def test_keyword_search_exact_match(self):
        """Test keyword search with exact match"""
        print_test("Keyword search with exact match")

        response = requests.post(
            f"{self.base_url}/search/keyword",
            json={
                "query": "Google's",
                "top_k": 5,
                "exact_match": True
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            result_count = len(data["results"])
            print_success(f"Found {result_count} results with exact match")
            return True  # Even 0 results is success (special chars handled)
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def test_structured_keyword_search(self):
        """Test structured keyword search (unified endpoint)"""
        print_test("Structured keyword search")

        response = requests.post(
            f"{self.base_url}/search/keyword",
            json={
                "required_keywords": ["learning"],
                "optional_keywords": ["machine", "deep"],
                "top_k": 5,
                "highlight": True
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            result_count = len(data["results"])
            print_success(f"Found {result_count} results (structured mode)")
            return result_count > 0
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def test_keyword_search_validation(self):
        """Test keyword search mode validation"""
        print_test("Keyword search mode validation")

        # Test 1: No query or keywords (should fail)
        response = requests.post(
            f"{self.base_url}/search/keyword",
            json={
                "top_k": 5
            },
            timeout=30
        )

        if response.status_code != 422:  # Pydantic validation error
            print_error(f"Expected validation error (422), got: {response.status_code}")
            return False

        print_success("Correctly rejected request with no query or keywords")

        # Test 2: Mixed mode (should fail)
        response = requests.post(
            f"{self.base_url}/search/keyword",
            json={
                "query": "machine learning",
                "required_keywords": ["learning"],
                "top_k": 5
            },
            timeout=30
        )

        if response.status_code != 422:  # Pydantic validation error
            print_error(f"Expected validation error (422), got: {response.status_code}")
            return False

        print_success("Correctly rejected mixed mode request")
        return True

    def test_remove_document(self):
        """Test document removal"""
        print_test("Remove document")

        if not self.document_ids:
            print_warning("No documents to remove")
            return True

        doc_id = self.document_ids[0]
        response = requests.post(
            f"{self.base_url}/documents/remove",
            json={"document_id": doc_id},
            timeout=30
        )

        if response.status_code == 200:
            print_success(f"Removed document {doc_id[:8]}...")
            return True
        else:
            print_error(f"Failed: {response.status_code} - {response.text}")
            return False

    def cleanup(self):
        """Clean up test files and documents"""
        print_test("Cleanup")

        # Remove remaining documents
        for doc_id in self.document_ids[1:]:  # Skip first (already removed)
            try:
                requests.post(
                    f"{self.base_url}/documents/remove",
                    json={"document_id": doc_id},
                    timeout=10
                )
            except:
                pass

        # Remove test files
        if hasattr(self, 'test_dir'):
            for file in self.test_files:
                try:
                    file.unlink()
                except:
                    pass
            try:
                self.test_dir.rmdir()
            except:
                pass

        print_success("Cleanup complete")

    def run_all_tests(self):
        """Run all API tests"""
        print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BLUE}RAG Anywhere API Test Suite{Colors.END}")
        print(f"{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"Testing server at: {self.base_url}")

        tests = [
            ("Setup", self.setup_test_files),
            ("Single Document Add", self.test_single_document_add),
            ("Batch Document Add", self.test_batch_document_add),
            ("List Documents", self.test_list_documents),
            ("Semantic Search", self.test_semantic_search),
            ("Keyword Search (Free-form)", self.test_keyword_search),
            ("Keyword Search (Exact Match)", self.test_keyword_search_exact_match),
            ("Keyword Search (Structured)", self.test_structured_keyword_search),
            ("Keyword Search (Validation)", self.test_keyword_search_validation),
            ("Remove Document", self.test_remove_document),
        ]

        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, result))
            except Exception as e:
                print_error(f"Exception in {name}: {e}")
                results.append((name, False))

        # Cleanup
        try:
            self.cleanup()
        except:
            pass

        # Summary
        print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BLUE}Test Summary{Colors.END}")
        print(f"{Colors.BLUE}{'='*60}{Colors.END}")

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for name, result in results:
            status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
            print(f"  {status} - {name}")

        print(f"\n{Colors.BLUE}Total: {passed}/{total} tests passed{Colors.END}")

        if passed == total:
            print(f"{Colors.GREEN}All tests passed!{Colors.END}")
            return 0
        else:
            print(f"{Colors.RED}{total - passed} test(s) failed{Colors.END}")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Test RAG Anywhere API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()

    tester = APITester(args.url)
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())