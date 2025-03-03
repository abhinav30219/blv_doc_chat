"""
Test script for LightRAGManager to verify multi-document querying.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Union

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system.lightrag_manager import LightRAGManager
from utils import logger

def test_lightrag_manager():
    """Test the LightRAGManager class."""
    try:
        print("Initializing LightRAGManager...")
        rag_manager = LightRAGManager()
        
        # Check if list_documents method works
        print("\nTesting list_documents method...")
        try:
            documents = rag_manager.list_documents()
            print(f"Found {len(documents)} documents:")
            for doc in documents:
                print(f"- {doc['title']} (ID: {doc['id']})")
        except Exception as e:
            print(f"Error in list_documents: {str(e)}")
            documents = []
        
        # If there are at least two documents, test querying across multiple documents
        if len(documents) >= 2:
            print("\nTesting query across multiple documents...")
            document_ids = [doc['id'] for doc in documents[:2]]
            print(f"Querying across documents: {document_ids}")
            
            try:
                # Test query with multiple document IDs
                result = rag_manager.query(
                    "What is this document about?",
                    document_id=document_ids
                )
                
                print("\nQuery result:")
                print(f"Query: {result['query']}")
                print(f"Answer: {result['answer'][:200]}..." if len(result['answer']) > 200 else result['answer'])
                print(f"Document IDs: {result['document_ids']}")
                print(f"Mode: {result['mode']}")
            except Exception as e:
                print(f"Error in query: {str(e)}")
        else:
            print("\nNot enough documents to test multi-document querying.")
            print("Please upload at least two documents first.")
    except Exception as e:
        print(f"Error initializing LightRAGManager: {str(e)}")

if __name__ == "__main__":
    test_lightrag_manager()
