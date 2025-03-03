"""
LightRAG manager for storing and retrieving document content.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
import numpy as np
import openai
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LightRAG
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from config import (
    RAG_WORKING_DIR,
    OPENAI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_TOKEN_SIZE,
    CHUNK_OVERLAP_TOKEN_SIZE,
    OPENAI_FILE_ENABLED,
    OPENAI_ASSISTANT_MODEL
)
from utils import logger, pipeline_logger, PipelineTimer, save_json, load_json
from rag_system.openai_file_manager import OpenAIFileManager

# Fallback SimpleRAG implementation
class SimpleRAG:
    """Enhanced in-memory RAG implementation with embeddings."""
    
    def __init__(self, working_dir: str):
        """Initialize the SimpleRAG."""
        self.working_dir = working_dir
        self.documents = {}
        self.chunks = {}
        self.embeddings = {}
        self.client = openai.OpenAI()
        
        # Create embeddings directory if it doesn't exist
        self.embeddings_dir = os.path.join(working_dir, "embeddings")
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        
        # Load existing embeddings if available
        self._load_embeddings()
    
    def _load_embeddings(self) -> None:
        """Load embeddings from disk."""
        embeddings_path = os.path.join(self.embeddings_dir, "embeddings.json")
        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'r') as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", {})
                    # Convert string keys back to tuples
                    self.embeddings = {}
                    for k, v in data.get("embeddings", {}).items():
                        try:
                            key = tuple(eval(k))
                            self.embeddings[key] = np.array(v)
                        except Exception as e:
                            logger.error(f"Error converting embedding key {k}: {e}")
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
    
    def _save_embeddings(self) -> None:
        """Save embeddings to disk."""
        embeddings_path = os.path.join(self.embeddings_dir, "embeddings.json")
        try:
            # Convert tuple keys to strings for JSON serialization
            serializable_embeddings = {
                str(k): v.tolist() for k, v in self.embeddings.items()
            }
            with open(embeddings_path, 'w') as f:
                json.dump({
                    "chunks": self.chunks,
                    "embeddings": serializable_embeddings
                }, f)
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of chunks with metadata
        """
        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        current_chunk_size = 0
        
        for i, para in enumerate(paragraphs):
            # If adding this paragraph would exceed chunk size, save current chunk and start a new one
            if current_chunk_size + len(para) > chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "start_idx": max(0, i - current_chunk_size // 200),  # Approximate paragraph index
                    "end_idx": i
                })
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                current_chunk_size = len(current_chunk)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_chunk_size += len(para) + 2  # +2 for the newlines
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "start_idx": max(0, len(paragraphs) - current_chunk_size // 200),
                "end_idx": len(paragraphs)
            })
        
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using OpenAI's embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        # Truncate text if too long
        if len(text) > 8000:
            text = text[:8000]
            
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        embedding = response.data[0].embedding
        return np.array(embedding)
    
    def insert(self, text: str, source_id: str) -> None:
        """
        Insert a document into the RAG system.
        
        Args:
            text: Document text
            source_id: Document ID
        """
        # Store original document
        self.documents[source_id] = text
        
        # Chunk document
        chunks = self._chunk_text(text)
        
        # Store chunks with source ID
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}_{i}"
            self.chunks[chunk_id] = {
                "text": chunk["text"],
                "source_id": source_id,
                "chunk_index": i,
                "metadata": {
                    "start_idx": chunk["start_idx"],
                    "end_idx": chunk["end_idx"]
                }
            }
            
            # Generate embedding for chunk
            try:
                embedding = self._get_embedding(chunk["text"])
                self.embeddings[(source_id, i)] = embedding
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i} of document {source_id}: {e}")
        
        # Save embeddings
        self._save_embeddings()
    
    def _search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using embeddings.
        
        Args:
            query_text: Query text
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with scores
        """
        # Get query embedding
        query_embedding = self._get_embedding(query_text)
        
        # Calculate similarity scores
        scores = {}
        for (source_id, chunk_idx), embedding in self.embeddings.items():
            chunk_id = f"{source_id}_{chunk_idx}"
            if chunk_id in self.chunks:
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                scores[chunk_id] = similarity
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k results
        results = []
        for chunk_id, score in sorted_scores[:top_k]:
            chunk = self.chunks[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "source_id": chunk["source_id"],
                "score": float(score),
                "metadata": chunk["metadata"]
            })
        
        return results
    
    def query(self, query_text: str, document_id: Optional[Union[str, List[str]]] = None, param: Any = None) -> str:
        """
        Query the RAG system.
        
        Args:
            query_text: Query text
            document_id: Optional document ID or list of document IDs to restrict search
            param: Optional parameters
            
        Returns:
            Response text
        """
        # If no documents, return empty response
        if not self.documents:
            logger.debug("SimpleRAG: No documents available to search")
            return "No documents available to search."
        
        # Convert single document_id to list for consistent handling
        document_ids = []
        if document_id:
            if isinstance(document_id, str):
                document_ids = [document_id]
            else:
                document_ids = document_id
            
            logger.debug(f"SimpleRAG: Restricting search to {len(document_ids)} documents")
        
        # Search for relevant chunks
        logger.debug(f"SimpleRAG: Searching for chunks relevant to: {query_text}")
        results = self._search(query_text)
        
        # Filter results by document_id if provided
        if document_ids:
            filtered_results = [r for r in results if r["source_id"] in document_ids]
            logger.debug(f"SimpleRAG: Filtered from {len(results)} to {len(filtered_results)} chunks based on document IDs")
            results = filtered_results
        
        logger.debug(f"SimpleRAG: Found {len(results)} relevant chunks")
        for i, r in enumerate(results):
            logger.debug(f"SimpleRAG: Chunk {i+1}: Score={r['score']:.4f}, Source={r['source_id']}, Text={r['text'][:100]}..." if len(r['text']) > 100 else r['text'])
        
        if not results:
            # If no results, use a fallback approach
            if document_ids:
                # If document IDs were provided but no results found, return a message
                logger.debug("SimpleRAG: No relevant chunks found in specified documents")
                return "I couldn't find relevant information in the selected documents. Please try rephrasing your question or selecting different documents."
            else:
                # If no document IDs were provided, use the first document as fallback
                logger.debug("SimpleRAG: No relevant chunks found, using first document as fallback")
                context = next(iter(self.documents.values()))
                logger.debug(f"SimpleRAG: Fallback context ({len(context)} chars): {context[:200]}..." if len(context) > 200 else context)
        else:
            # Combine relevant chunks
            context = "\n\n".join([f"[From document {r['source_id']}] [Relevance: {r['score']:.2f}] {r['text']}" for r in results])
            logger.debug(f"SimpleRAG: Combined context ({len(context)} chars): {context[:200]}..." if len(context) > 200 else context)
        
        # Use OpenAI to generate a response
        logger.debug("SimpleRAG: Generating response using OpenAI")
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided context.
                 Be specific and detailed in your answers, citing information from the context.
                 If the context doesn't contain the information needed to answer the question, say so clearly."""},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            ],
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        logger.debug(f"SimpleRAG: Generated response: {result[:200]}..." if len(result) > 200 else result)
        return result


class LightRAGManager:
    """
    Manager for LightRAG to store and retrieve document content.
    """
    
    def __init__(self):
        """Initialize the LightRAG manager."""
        # Ensure working directory exists
        if not os.path.exists(RAG_WORKING_DIR):
            os.makedirs(RAG_WORKING_DIR)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize OpenAI file manager if enabled
        self.file_manager = None
        if OPENAI_FILE_ENABLED:
            try:
                self.file_manager = OpenAIFileManager()
                logger.info("OpenAI file manager initialized")
            except Exception as e:
                logger.error(f"Error initializing OpenAI file manager: {e}")
                logger.warning("OpenAI file integration disabled")
        
        # Initialize LightRAG
        try:
            # Define embedding function
            async def embedding_func(texts: list[str]) -> np.ndarray:
                return await openai_embed(
                    texts,
                    model=EMBEDDING_MODEL,
                    api_key=OPENAI_API_KEY
                )
            
            # Define LLM function
            async def llm_model_func(
                prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
            ) -> str:
                return await openai_complete_if_cache(
                    LLM_MODEL,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=OPENAI_API_KEY,
                    **kwargs
                )
            
            # Get embedding dimension
            async def get_embedding_dim():
                test_text = ["This is a test sentence."]
                embedding = await embedding_func(test_text)
                embedding_dim = embedding.shape[1]
                return embedding_dim
            
            # Run in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            embedding_dimension = loop.run_until_complete(get_embedding_dim())
            
            # Initialize LightRAG
            self.rag = LightRAG(
                working_dir=RAG_WORKING_DIR,
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dimension,
                    max_token_size=8192,
                    func=embedding_func
                ),
                chunk_token_size=CHUNK_TOKEN_SIZE,
                chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
                tiktoken_model_name="gpt-4"  # Use a model name that tiktoken recognizes
            )
            
            logger.info("LightRAG initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LightRAG: {e}")
            # Fallback to SimpleRAG
            logger.warning("Falling back to SimpleRAG")
            self.rag = SimpleRAG(working_dir=RAG_WORKING_DIR)
        
        # Document metadata storage
        self.document_metadata = {}
        self._load_document_metadata()
        
        logger.info("LightRAGManager initialized")
    
    def _load_document_metadata(self) -> None:
        """Load document metadata from file."""
        metadata_path = os.path.join(RAG_WORKING_DIR, "document_metadata.json")
        if os.path.exists(metadata_path):
            try:
                self.document_metadata = load_json(metadata_path)
                logger.info(f"Loaded metadata for {len(self.document_metadata)} documents")
            except Exception as e:
                logger.error(f"Error loading document metadata: {e}")
                self.document_metadata = {}
    
    def _save_document_metadata(self) -> None:
        """Save document metadata to file."""
        metadata_path = os.path.join(RAG_WORKING_DIR, "document_metadata.json")
        try:
            save_json(self.document_metadata, metadata_path)
            logger.info(f"Saved metadata for {len(self.document_metadata)} documents")
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")
    
    def add_document(self, document_content: Dict[str, Any]) -> str:
        """
        Add a document to the RAG system.
        
        Args:
            document_content: Document content from DocumentProcessor
            
        Returns:
            Document ID
        """
        with PipelineTimer("RAG Document Indexing"):
            # Generate document ID
            import uuid
            document_id = str(uuid.uuid4())
            
            title = document_content['metadata']['title']
            if pipeline_logger:
                pipeline_logger.info(f"Adding document to RAG: {title} (ID: {document_id})")
            
            # Store document metadata
            self.document_metadata[document_id] = document_content["metadata"]
            self._save_document_metadata()
            
            # Prepare text for RAG
            with PipelineTimer("Text Chunking"):
                elements = document_content["elements"]
                text_chunks = []
                
                # Process each element
                for element in elements:
                    # Add element text
                    if "text" in element and element["text"]:
                        # Add location metadata to the text
                        element_type = element.get("type", "text")
                        metadata_str = ""
                        
                        if "metadata" in element:
                            # Extract location information from metadata
                            if "page_number" in element["metadata"]:
                                metadata_str += f" [Page: {element['metadata']['page_number']}]"
                            if "section_name" in element["metadata"]:
                                metadata_str += f" [Section: {element['metadata']['section_name']}]"
                        
                        # Add element with metadata
                        text_chunks.append(f"{element_type}{metadata_str}: {element['text']}")
                
                # Process images
                for image in document_content.get("images", []):
                    # Add image description with location
                    location_str = ""
                    if "page" in image:
                        location_str += f" [Page: {image['page']}]"
                    if "paragraph" in image:
                        location_str += f" [Paragraph: {image['paragraph']}]"
                    
                    # Add image description
                    text_chunks.append(f"Image{location_str}: {image['description']}")
                    
                    # Add OCR text if available
                    if "ocr_text" in image and image["ocr_text"]:
                        text_chunks.append(f"Image OCR Text{location_str}: {image['ocr_text']}")
                
                # Add document structure information
                sections = document_content["metadata"]["sections"]
                if sections:
                    structure_text = "Document Structure:\n"
                    for section in sections:
                        level_prefix = "  " * (section.get("level", 1) - 1)
                        structure_text += f"{level_prefix}- {section.get('title', 'Untitled Section')} ({section.get('start_position', 'unknown position')}): {section.get('content_summary', 'No summary')}\n"
                    text_chunks.append(structure_text)
                
                # Combine text chunks
                full_text = "\n\n".join(text_chunks)
                
                if pipeline_logger:
                    pipeline_logger.info(f"Created {len(text_chunks)} text chunks for indexing")
            
            # Add document to RAG
            try:
                with PipelineTimer("Vector Embedding Generation"):
                    # Check if we're using LightRAG or SimpleRAG
                    if hasattr(self.rag, 'insert') and not hasattr(self.rag, 'ainsert'):
                        # SimpleRAG
                        self.rag.insert(full_text, source_id=document_id)
                    else:
                        # LightRAG
                        # Use document ID as the ID parameter
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.rag.ainsert(full_text, ids=document_id))
                    
                    if pipeline_logger:
                        pipeline_logger.info(f"Document indexed in vector database")
                
                # Upload document to OpenAI if file manager is available
                if self.file_manager:
                    with PipelineTimer("OpenAI File Upload"):
                        try:
                            # Create a temporary file with the document content
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                                temp_file.write(full_text)
                                temp_file_path = temp_file.name
                            
                            # Upload file to OpenAI
                            file_id = self.file_manager.upload_file(temp_file_path, document_id)
                            
                            # Create assistant for the document
                            assistant_id = self.file_manager.create_assistant_with_file(
                                document_id,
                                name=f"Document Assistant - {document_content['metadata']['title']}",
                                instructions="You are a helpful assistant that answers questions about this document. Be specific and detailed in your answers, citing information from the document.",
                                model=OPENAI_ASSISTANT_MODEL
                            )
                            
                            # Update document metadata with file and assistant IDs
                            self.document_metadata[document_id]["openai_file_id"] = file_id
                            self.document_metadata[document_id]["openai_assistant_id"] = assistant_id
                            self._save_document_metadata()
                            
                            # Clean up temporary file
                            import os
                            os.unlink(temp_file_path)
                            
                            if pipeline_logger:
                                pipeline_logger.info(f"Document uploaded to OpenAI and assistant created")
                        except Exception as e:
                            logger.error(f"Error uploading document to OpenAI: {e}")
                            # Continue even if OpenAI upload fails
                
                if pipeline_logger:
                    pipeline_logger.info(f"Document processing complete: {title}")
                
                return document_id
            except Exception as e:
                logger.error(f"Error adding document to RAG: {e}")
                if pipeline_logger:
                    pipeline_logger.info(f"Document processing failed: {title}")
                raise
    
    def query(self, query_text: str, document_id: Optional[Union[str, List[str]]] = None, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query_text: Query text
            document_id: Optional document ID or list of document IDs to restrict search
            mode: RAG query mode ("naive", "local", "global", or "hybrid")
            
        Returns:
            Dict containing query result
        """
        with PipelineTimer("RAG Query Processing"):
            if pipeline_logger:
                pipeline_logger.info(f"Query received: '{query_text[:50]}{'...' if len(query_text) > 50 else ''}'")
            
            # Convert single document_id to list for consistent handling
            document_ids = []
            if document_id:
                if isinstance(document_id, str):
                    document_ids = [document_id]
                else:
                    document_ids = document_id
                
                if pipeline_logger:
                    if len(document_ids) == 1:
                        pipeline_logger.info(f"Restricting search to document: {document_ids[0]}")
                    else:
                        pipeline_logger.info(f"Restricting search to {len(document_ids)} documents")
            
            # Analyze query to detect if it's a location query
            with PipelineTimer("Query Analysis"):
                is_location_query, location_info = self._analyze_query_for_location(query_text)
                
                if is_location_query and pipeline_logger:
                    pipeline_logger.info(f"Detected location query: {location_info['type']} {location_info['identifier']}")
            
            if is_location_query:
                return self._handle_location_query(query_text, location_info, document_ids[0] if document_ids else None)
            
            # Regular content query
            try:
                # Try to use OpenAI file manager if available and a single document_id is provided
                if self.file_manager and len(document_ids) == 1:
                    single_doc_id = document_ids[0]
                    if single_doc_id in self.document_metadata:
                        # Check if we have an assistant ID for this document
                        if "openai_assistant_id" in self.document_metadata[single_doc_id]:
                            with PipelineTimer("OpenAI Assistant Query"):
                                if pipeline_logger:
                                    pipeline_logger.info(f"Using OpenAI assistant for document")
                                
                                # Query the document using OpenAI assistant
                                assistant_result = self.file_manager.query_document(single_doc_id, query_text)
                                
                                if assistant_result:
                                    if pipeline_logger:
                                        pipeline_logger.info(f"OpenAI assistant returned answer ({len(assistant_result)} chars)")
                                    return {
                                        "query": query_text,
                                        "answer": assistant_result,
                                        "document_id": single_doc_id,
                                        "mode": "openai_assistant",
                                        "is_location_query": False
                                    }
                                else:
                                    if pipeline_logger:
                                        pipeline_logger.info(f"OpenAI assistant query failed, falling back to RAG")
                
                # Fall back to RAG if OpenAI file manager is not available or failed
                with PipelineTimer("Vector Search"):
                    # Check if we're using LightRAG or SimpleRAG
                    if hasattr(self.rag, 'query'):
                        # SimpleRAG
                        if pipeline_logger:
                            pipeline_logger.info(f"Using SimpleRAG for retrieval")
                        
                        # Pass document_ids to SimpleRAG query
                        if document_ids:
                            if pipeline_logger:
                                pipeline_logger.info(f"Querying SimpleRAG with document IDs: {document_ids}")
                            
                            # Pass document_ids to SimpleRAG query
                            result = self.rag.query(query_text, document_ids)
                        else:
                            # Query all documents
                            if pipeline_logger:
                                pipeline_logger.info("Querying SimpleRAG with all documents")
                            
                            result = self.rag.query(query_text)
                    else:
                        # LightRAG
                        if pipeline_logger:
                            pipeline_logger.info(f"Using LightRAG for retrieval (mode: {mode})")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Create a param with context logging enabled
                        query_param = QueryParam(mode=mode, only_need_context=False)
                        
                        # If document_ids are provided, add them to the query parameters
                        if document_ids:
                            # Execute query for each document and combine results
                            all_results = []
                            for doc_id in document_ids:
                                # Get document metadata for logging
                                doc_metadata = self.get_document_metadata(doc_id)
                                doc_title = doc_metadata["title"] if doc_metadata else "Unknown document"
                                
                                if pipeline_logger:
                                    pipeline_logger.info(f"Querying document: {doc_title} (ID: {doc_id})")
                                
                                try:
                                    # Execute query with document ID filter
                                    # Ensure param is a QueryParam object, not a list
                                    if isinstance(query_param, list):
                                        # If it's a list, create a new QueryParam
                                        safe_param = QueryParam(mode=mode, only_need_context=False)
                                    else:
                                        safe_param = query_param
                                        
                                    doc_result = loop.run_until_complete(
                                        self.rag.aquery(
                                            query_text, 
                                            param=safe_param,
                                            ids=doc_id  # Filter by document ID
                                        )
                                    )
                                    
                                    # Log result length for debugging
                                    if pipeline_logger:
                                        result_length = len(doc_result) if doc_result else 0
                                        pipeline_logger.info(f"Document {doc_title} returned {result_length} characters")
                                    
                                    # Add document title to result for clarity
                                    if doc_result and len(doc_result.strip()) > 0:
                                        doc_result = f"[From {doc_title}]:\n{doc_result}"
                                        all_results.append(doc_result)
                                    else:
                                        if pipeline_logger:
                                            pipeline_logger.info(f"Document {doc_title} returned no results for query: '{query_text[:50]}{'...' if len(query_text) > 50 else ''}'")
                                except Exception as e:
                                    logger.error(f"Error querying document {doc_title}: {e}")
                                    if pipeline_logger:
                                        pipeline_logger.info(f"Error querying document {doc_title}: {e}")
                            
                            # Combine results
                            if all_results:
                                result = "\n\n".join(all_results)
                                if pipeline_logger:
                                    pipeline_logger.info(f"Combined results from {len(all_results)} documents")
                            else:
                                result = "I couldn't find relevant information in the selected documents. Please try rephrasing your question or selecting different documents."
                                if pipeline_logger:
                                    pipeline_logger.info(f"No results found in any of the selected documents")
                        else:
                            # Execute query without document ID filter
                            result = loop.run_until_complete(
                                self.rag.aquery(query_text, param=query_param)
                            )
                    
                    if pipeline_logger:
                        pipeline_logger.info(f"Retrieved answer ({len(result)} chars)")
                
                # Format result
                return {
                    "query": query_text,
                    "answer": result,
                    "document_ids": document_ids,
                    "mode": mode,
                    "is_location_query": False
                }
            except Exception as e:
                logger.error(f"Error querying RAG: {e}")
                if pipeline_logger:
                    pipeline_logger.info(f"Query processing failed: {str(e)}")
                return {
                    "query": query_text,
                    "answer": f"Error: {str(e)}",
                    "document_ids": document_ids,
                    "mode": mode,
                    "is_location_query": False
                }
    
    def _analyze_query_for_location(self, query_text: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze query to detect if it's asking about a specific location in the document.
        
        Args:
            query_text: Query text
            
        Returns:
            Tuple of (is_location_query, location_info)
        """
        try:
            # Use LLM to analyze query
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You analyze document queries to determine if they are asking about specific locations in a document.
                     If the query is asking about content at a specific location (e.g., "What's on page 5?", "What's in section 3?"),
                     extract the location information and return it as JSON.
                     If it's a regular content query, indicate that it's not a location query.
                     
                     Return your analysis in JSON format with the following structure:
                     {
                       "is_location_query": boolean,
                       "location_info": {
                         "type": "page" or "section" or "paragraph" etc.,
                         "identifier": the page number, section name, etc.
                       }
                     }
                     
                     Your response must be valid JSON."""},
                    {"role": "user", "content": f"Analyze this query: {query_text}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Check if it's a location query
            is_location_query = analysis.get("is_location_query", False)
            location_info = analysis.get("location_info", None)
            
            return is_location_query, location_info
        except Exception as e:
            logger.error(f"Error analyzing query for location: {e}")
            return False, None
            
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        return self.document_metadata.get(document_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the RAG system.
        
        Returns:
            List of document metadata
        """
        return [
            {"id": doc_id, **metadata}
            for doc_id, metadata in self.document_metadata.items()
        ]
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the RAG system.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        if document_id in self.document_metadata:
            # Delete OpenAI file and assistant if available
            if self.file_manager:
                try:
                    # Check if we have OpenAI file and assistant IDs
                    if "openai_file_id" in self.document_metadata[document_id]:
                        # Delete assistant first
                        if "openai_assistant_id" in self.document_metadata[document_id]:
                            self.file_manager.delete_assistant(document_id)
                            logger.info(f"Deleted OpenAI assistant for document {document_id}")
                        
                        # Delete file
                        self.file_manager.delete_file(document_id)
                        logger.info(f"Deleted OpenAI file for document {document_id}")
                except Exception as e:
                    logger.error(f"Error deleting OpenAI file and assistant: {e}")
                    # Continue even if OpenAI deletion fails
            
            # Delete from metadata
            del self.document_metadata[document_id]
            self._save_document_metadata()
            
            # Note: LightRAG doesn't currently support deleting documents
            # This is a limitation of the current version
            logger.warning("Document deleted from metadata, but content remains in LightRAG index")
            
            return True
        return False
    
    def _handle_location_query(self, query_text: str, location_info: Dict[str, Any], document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle a query about a specific location in the document.
        
        Args:
            query_text: Original query text
            location_info: Location information extracted from query
            document_id: Optional document ID to restrict search
            
        Returns:
            Dict containing query result
        """
        logger.info(f"Handling location query: {location_info}")
        
        try:
            # Extract location type and identifier
            location_type = location_info.get("type", "").lower()
            location_identifier = location_info.get("identifier", "")
            
            # Create a specific search pattern based on location type
            search_pattern = ""
            if location_type == "page":
                search_pattern = f"[Page: {location_identifier}]"
            elif location_type == "section":
                search_pattern = f"[Section: {location_identifier}]"
            elif location_type == "paragraph":
                search_pattern = f"[Paragraph: {location_identifier}]"
            else:
                # Generic pattern
                search_pattern = f"{location_type}: {location_identifier}"
            
            # Create a search query that specifically looks for the location pattern
            search_query = f"Find content with the pattern '{search_pattern}'"
            
            # Check if we're using LightRAG or SimpleRAG
            if hasattr(self.rag, 'chunks'):
                # SimpleRAG
                # Get all chunks that might contain the location
                relevant_chunks = []
                for chunk_id, chunk in self.rag.chunks.items():
                    if search_pattern.lower() in chunk["text"].lower():
                        relevant_chunks.append(chunk["text"])
                
                logger.debug(f"Found {len(relevant_chunks)} chunks matching location pattern '{search_pattern}'")
                
                # If no chunks found with the exact pattern, fall back to regular search
                if not relevant_chunks:
                    # Use LLM to generate a search query based on the location
                    logger.debug(f"No chunks found with pattern '{search_pattern}', generating search query")
                    response = self.client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": """You generate search queries for a RAG system based on location information.
                             The goal is to find content at specific locations in documents."""},
                            {"role": "user", "content": f"Generate a search query to find content at this location: {json.dumps(location_info)}"}
                        ],
                        max_tokens=100
                    )
                    
                    search_query = response.choices[0].message.content
                    logger.debug(f"Generated search query for location: {search_query}")
                    # Pass document_id to SimpleRAG query if provided
                    if document_id:
                        search_result = self.rag.query(search_query, document_id)
                    else:
                        search_result = self.rag.query(search_query)
                    logger.debug(f"Search result: {search_result[:200]}..." if len(search_result) > 200 else search_result)
                else:
                    # Combine relevant chunks
                    context = "\n\n".join(relevant_chunks)
                    logger.debug(f"Location query context ({len(context)} chars): {context[:200]}..." if len(context) > 200 else context)
                    
                    # Use OpenAI to generate a response based on the relevant chunks
                    response = self.client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": f"""You are answering a question about content at a specific location in a document.
                             The location is: {location_type} {location_identifier}.
                             Focus only on the content that appears at this specific location.
                             Be clear and specific about what information is found at this location."""},
                            {"role": "user", "content": f"Question: {query_text}\n\nContent at the specified location:\n{context}"}
                        ],
                        max_tokens=1000
                    )
                    
                    search_result = response.choices[0].message.content
                    logger.debug(f"Generated response for location query: {search_result[:200]}..." if len(search_result) > 200 else search_result)
            else:
                # LightRAG
                # Create a search query that specifically looks for the location pattern
                location_query = f"Find content at {location_type} {location_identifier}"
                logger.debug(f"Location query for LightRAG: {location_query}")
                
                # Execute query
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                search_result = loop.run_until_complete(
                    self.rag.aquery(location_query, param=QueryParam(mode="naive"))
                )
                logger.debug(f"Location query result: {search_result[:200]}..." if len(search_result) > 200 else search_result)
            
            # Use LLM to format the result as an answer to the original query
            final_response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You answer questions about specific locations in documents.
                     Format your response to clearly indicate what content is found at the requested location.
                     For blind and low vision users, be very explicit about the location being referenced."""},
                    {"role": "user", "content": f"Original query: {query_text}\nLocation: {json.dumps(location_info)}\nContent found: {search_result}"}
                ],
                max_tokens=1000
            )
            
            answer = final_response.choices[0].message.content
            
            return {
                "query": query_text,
                "answer": answer,
                "mode": "location",
                "is_location_query": True,
                "location_info": location_info
            }
        except Exception as e:
            logger.error(f"Error handling location query: {e}")
            return {
                "query": query_text,
                "answer": f"Error handling location query: {str(e)}",
                "mode": "location",
                "is_location_query": True,
                "location_info": location_info
            }
