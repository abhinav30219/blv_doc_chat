"""
Multi-agent system for coordinating document analysis and question answering.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import openai

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY,
    LLM_MODEL
)
from utils import logger, pipeline_logger, PipelineTimer
from rag_system.lightrag_manager import LightRAGManager

class MultiAgentSystem:
    """
    Multi-agent system for coordinating document analysis and question answering.
    """
    
    def __init__(self, rag_manager: LightRAGManager):
        """
        Initialize the multi-agent system.
        
        Args:
            rag_manager: LightRAG manager for document retrieval
        """
        self.rag_manager = rag_manager
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.conversation_history = []
        self.max_queries_per_document = 3  # Maximum number of queries to generate per document
        logger.info("MultiAgentSystem initialized")
    
    def process_query(self, query: str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a user query using the multi-agent system.
        
        Args:
            query: User query
            document_ids: Optional list of document IDs to include in search
            
        Returns:
            Dict containing the response
        """
        with PipelineTimer("Multi-Agent Query Processing"):
            if pipeline_logger:
                pipeline_logger.info(f"Query received by multi-agent system: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            # Add query to conversation history
            self.conversation_history.append({"role": "user", "content": query})
            
            # Analyze query to determine the best approach
            with PipelineTimer("Query Type Analysis"):
                query_analysis = self._analyze_query(query)
                if pipeline_logger:
                    pipeline_logger.info(f"Query classified as: {query_analysis['query_type']}")
            
            # Get response based on query analysis
            if query_analysis["query_type"] == "document_question":
                # Document-related question
                response = self._handle_document_question(query, document_ids, query_analysis)
            elif query_analysis["query_type"] == "conversation_question":
                # Question about the conversation
                response = self._handle_conversation_question(query, query_analysis)
            elif query_analysis["query_type"] == "system_command":
                # System command
                response = self._handle_system_command(query, query_analysis)
            else:
                # General question
                response = self._handle_general_question(query, query_analysis)
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response["answer"]})
            
            if pipeline_logger:
                pipeline_logger.info(f"Generated response ({len(response['answer'])} chars)")
            
            # Return response
            return response
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query to determine the best approach.
        
        Args:
            query: User query
            
        Returns:
            Dict containing query analysis
        """
        logger.info(f"Analyzing query: {query}")
        
        try:
            # Use LLM to analyze query
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You analyze user queries to determine the best approach for answering them.
                     Categorize the query into one of the following types:
                     - document_question: Question about a document
                     - conversation_question: Question about the conversation history
                     - system_command: Command to the system
                     - general_question: General question not related to a document
                     
                     Also extract any relevant parameters from the query.
                     
                     Return your analysis in JSON format with the following structure:
                     {
                       "query_type": "document_question" or "conversation_question" or "system_command" or "general_question",
                       "command": "command_name" (if query_type is system_command),
                       "parameters": {
                         // Any relevant parameters extracted from the query
                       }
                     }
                     
                     Your response must be valid JSON."""},
                    {"role": "user", "content": f"Analyze this query: {query}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            analysis = json.loads(response.choices[0].message.content)
            logger.info(f"Query analysis: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {"query_type": "document_question"}
    
    def _handle_document_question(self, query: str, document_ids: Optional[List[str]], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a document-related question.
        
        Args:
            query: User query
            document_ids: Optional list of document IDs to include in search
            query_analysis: Query analysis
            
        Returns:
            Dict containing the response
        """
        with PipelineTimer("Document Question Processing"):
            if pipeline_logger:
                pipeline_logger.info(f"Processing document question")
            
            # Get document summaries for context
            document_summaries = self._get_document_summaries(document_ids)
            
            # Generate contextual queries based on conversation history and document content
            with PipelineTimer("Query Generation"):
                contextual_queries = self._generate_contextual_queries(
                    query, 
                    self.conversation_history, 
                    document_ids, 
                    document_summaries
                )
                
                if pipeline_logger:
                    pipeline_logger.info(f"Generated {len(contextual_queries)} contextual queries")
                    for i, q in enumerate(contextual_queries):
                        pipeline_logger.info(f"Query {i+1}: {q['query'][:50]}{'...' if len(q['query']) > 50 else ''}")
            
            # Execute all queries and collect results
            with PipelineTimer("Query Execution"):
                all_results = []
                for contextual_query in contextual_queries:
                    # Execute query against LightRAG
                    rag_response = self.rag_manager.query(
                        contextual_query["query"], 
                        contextual_query.get("document_id")
                    )
                    
                    # Add query metadata to response
                    result = {
                        "query": contextual_query["query"],
                        "answer": rag_response["answer"],
                        "document_id": contextual_query.get("document_id"),
                        "purpose": contextual_query.get("purpose", "general"),
                        "mode": rag_response.get("mode")
                    }
                    all_results.append(result)
                
                if pipeline_logger:
                    pipeline_logger.info(f"Executed {len(all_results)} queries")
            
            # Synthesize final response from all results
            with PipelineTimer("Response Synthesis"):
                final_response = self._synthesize_response(query, all_results)
                if pipeline_logger:
                    pipeline_logger.info(f"Synthesized final response")
            
            return final_response
    
    def _handle_conversation_question(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a question about the conversation.
        
        Args:
            query: User query
            query_analysis: Query analysis
            
        Returns:
            Dict containing the response
        """
        logger.info(f"Handling conversation question: {query}")
        
        try:
            # Use LLM to answer question about conversation
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-10:]  # Last 10 messages
            ])
            
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You answer questions about the conversation history.
                     Provide clear, concise answers based on the conversation context.
                     For blind and low vision users, be explicit about what was discussed."""},
                    {"role": "user", "content": f"Conversation context:\n{conversation_context}\n\nQuestion: {query}"}
                ],
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "query": query,
                "answer": answer,
                "query_type": "conversation_question"
            }
        except Exception as e:
            logger.error(f"Error handling conversation question: {e}")
            return {
                "query": query,
                "answer": f"Error handling conversation question: {str(e)}",
                "query_type": "conversation_question"
            }
    
    def _handle_system_command(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a system command.
        
        Args:
            query: User query
            query_analysis: Query analysis
            
        Returns:
            Dict containing the response
        """
        logger.info(f"Handling system command: {query}")
        
        # Extract command from query analysis
        command = query_analysis.get("command", "unknown")
        
        if command == "clear_history":
            # Clear conversation history
            self.conversation_history = []
            return {
                "query": query,
                "answer": "Conversation history cleared.",
                "query_type": "system_command",
                "command": command
            }
        elif command == "help":
            # Show help
            help_text = """
            Available commands:
            - clear_history: Clear conversation history
            - help: Show this help message
            
            You can ask questions about documents, the conversation, or general topics.
            You can select specific documents to include in your search.
            """
            return {
                "query": query,
                "answer": help_text,
                "query_type": "system_command",
                "command": command
            }
        elif command == "list_documents":
            # List available documents
            documents = self.rag_manager.list_documents()
            if documents:
                doc_list = "\n".join([f"- {doc['title']}" for doc in documents])
                return {
                    "query": query,
                    "answer": f"Available documents:\n{doc_list}",
                    "query_type": "system_command",
                    "command": command
                }
            else:
                return {
                    "query": query,
                    "answer": "No documents available.",
                    "query_type": "system_command",
                    "command": command
                }
        else:
            # Unknown command
            return {
                "query": query,
                "answer": f"Unknown command: {command}. Type 'help' for a list of available commands.",
                "query_type": "system_command",
                "command": command
            }
    
    def _handle_general_question(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a general question.
        
        Args:
            query: User query
            query_analysis: Query analysis
            
        Returns:
            Dict containing the response
        """
        logger.info(f"Handling general question: {query}")
        
        try:
            # Use LLM to answer general question
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-5:]  # Last 5 messages
            ])
            
            # Get available documents for context
            documents = self.rag_manager.list_documents()
            document_context = ""
            if documents:
                document_context = "Available documents:\n" + "\n".join([
                    f"- {doc['title']}" for doc in documents
                ])
            
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You answer general questions.
                     Provide clear, concise answers based on your knowledge.
                     If the question might be about a document, suggest that the user ask a document-specific question instead.
                     For blind and low vision users, be clear and descriptive in your responses."""},
                    {"role": "user", "content": f"Conversation context:\n{conversation_context}\n\nAvailable documents:\n{document_context}\n\nQuestion: {query}"}
                ],
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "query": query,
                "answer": answer,
                "query_type": "general_question"
            }
        except Exception as e:
            logger.error(f"Error handling general question: {e}")
            return {
                "query": query,
                "answer": f"Error handling general question: {str(e)}",
                "query_type": "general_question"
            }
    
    def _get_document_summaries(self, document_ids: Optional[List[str]]) -> Dict[str, str]:
        """
        Get summaries for the specified documents.
        
        Args:
            document_ids: List of document IDs to get summaries for
            
        Returns:
            Dict mapping document IDs to summaries
        """
        summaries = {}
        
        # If no document IDs provided, get all documents
        if not document_ids:
            documents = self.rag_manager.list_documents()
            document_ids = [doc["id"] for doc in documents]
        
        # Get metadata for each document
        for doc_id in document_ids:
            metadata = self.rag_manager.get_document_metadata(doc_id)
            if metadata:
                # Create a summary from metadata
                summary = f"Title: {metadata['title']}\n"
                
                # Add sections if available
                if "sections" in metadata and metadata["sections"]:
                    summary += "Sections:\n"
                    for section in metadata["sections"]:
                        section_title = section.get("title", "Untitled Section")
                        section_level = section.get("level", 1)
                        section_summary = section.get("content_summary", "No summary available")
                        summary += f"{'  ' * (section_level - 1)}- {section_title}: {section_summary}\n"
                
                summaries[doc_id] = summary
        
        return summaries
    
    def _generate_contextual_queries(
        self, 
        user_query: str, 
        conversation_history: List[Dict[str, Any]], 
        document_ids: Optional[List[str]], 
        document_summaries: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple contextual queries based on conversation history and document content.
        
        Args:
            user_query: The current user query
            conversation_history: Previous conversation messages
            document_ids: List of document IDs to include in search
            document_summaries: Summaries of documents
            
        Returns:
            List of generated queries with metadata
        """
        # If no document IDs provided, get all documents
        if not document_ids:
            documents = self.rag_manager.list_documents()
            document_ids = [doc["id"] for doc in documents]
        
        # Prepare conversation context (last 5 messages)
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_history
        ])
        
        # Prepare document context
        document_context = "\n\n".join([
            f"Document {i+1}:\n{document_summaries.get(doc_id, 'No summary available')}"
            for i, doc_id in enumerate(document_ids)
        ])
        
        # Generate queries using LLM
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You are a query generation assistant that creates multiple search queries based on a user's question, conversation history, and document context.
                     
                     For each document, generate up to 3 specific queries that:
                     1. Address different aspects of the user's question
                     2. Consider the conversation history for context
                     3. Are tailored to the specific content of each document
                     4. Include follow-up questions implied by the conversation
                     
                     Also generate 2-3 general queries that might connect information across documents.
                     
                     Format your response as a JSON array of query objects, where each object has:
                     - "query": The generated search query
                     - "document_id": The ID of the document this query targets (or null for general queries)
                     - "purpose": A brief description of what this query is trying to find
                     
                     Your response must be valid JSON."""},
                    {"role": "user", "content": f"User query: {user_query}\n\nConversation history:\n{conversation_context}\n\nDocument context:\n{document_context}\n\nDocument IDs: {document_ids}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse response
            queries_data = json.loads(response.choices[0].message.content)
            
            # Extract queries
            if "queries" in queries_data:
                queries = queries_data["queries"]
            else:
                # Fallback if the response doesn't have the expected structure
                queries = queries_data
            
            # Ensure we have at least one query
            if not queries:
                # Fallback to original query
                queries = [{
                    "query": user_query,
                    "document_id": None,
                    "purpose": "original query"
                }]
            
            # Limit number of queries per document
            filtered_queries = []
            doc_query_counts = {}
            
            for query in queries:
                doc_id = query.get("document_id")
                
                # Initialize count for this document if not already done
                if doc_id not in doc_query_counts:
                    doc_query_counts[doc_id] = 0
                
                # Add query if we haven't reached the limit for this document
                if doc_query_counts[doc_id] < self.max_queries_per_document:
                    filtered_queries.append(query)
                    doc_query_counts[doc_id] += 1
            
            return filtered_queries
            
        except Exception as e:
            logger.error(f"Error generating contextual queries: {e}")
            # Fallback to original query
            return [{
                "query": user_query,
                "document_id": None,
                "purpose": "original query"
            }]
    
    def _synthesize_response(self, original_query: str, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize a final response from multiple query results.
        
        Args:
            original_query: The original user query
            query_results: Results from multiple queries
            
        Returns:
            Dict containing the synthesized response
        """
        try:
            # Prepare context from all results
            results_context = ""
            for i, result in enumerate(query_results):
                doc_id = result.get("document_id", "unknown")
                purpose = result.get("purpose", "general")
                
                # Get document title if available
                doc_title = "unknown document"
                if doc_id:
                    metadata = self.rag_manager.get_document_metadata(doc_id)
                    if metadata:
                        doc_title = metadata.get("title", "unknown document")
                
                results_context += f"Result {i+1} (from {doc_title}, purpose: {purpose}):\n{result['answer']}\n\n"
            
            # Generate synthesized response
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You synthesize information from multiple search results to create a comprehensive answer.
                     
                     Create a coherent, well-structured response that:
                     1. Directly answers the user's question
                     2. Integrates information from all relevant results
                     3. Cites which documents provided which information
                     4. Resolves any contradictions between sources
                     5. Acknowledges if certain information was not found
                     
                     For blind and low vision users, be clear about the source of information."""},
                    {"role": "user", "content": f"Original query: {original_query}\n\nSearch results:\n{results_context}"}
                ],
                max_tokens=1000
            )
            
            synthesized_answer = response.choices[0].message.content
            
            # Return synthesized response
            return {
                "query": original_query,
                "answer": synthesized_answer,
                "original_results": query_results,
                "query_type": "document_question"
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            
            # Fallback to first result if available
            if query_results:
                return {
                    "query": original_query,
                    "answer": query_results[0]["answer"],
                    "query_type": "document_question"
                }
            else:
                return {
                    "query": original_query,
                    "answer": f"Error synthesizing response: {str(e)}",
                    "query_type": "document_question"
                }
