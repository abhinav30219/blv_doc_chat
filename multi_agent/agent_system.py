"""
Multi-agent system for coordinating document analysis and question answering.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import openai

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY,
    LLM_MODEL
)
from utils import logger
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
        logger.info("MultiAgentSystem initialized")
    
    def process_query(self, query: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query using the multi-agent system.
        
        Args:
            query: User query
            document_id: Optional document ID to restrict search
            
        Returns:
            Dict containing the response
        """
        logger.info(f"Processing query: {query}")
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Analyze query to determine the best approach
        query_analysis = self._analyze_query(query)
        
        # Get response based on query analysis
        if query_analysis["query_type"] == "document_question":
            # Document-related question
            response = self._handle_document_question(query, document_id, query_analysis)
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
    
    def _handle_document_question(self, query: str, document_id: Optional[str], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a document-related question.
        
        Args:
            query: User query
            document_id: Optional document ID to restrict search
            query_analysis: Query analysis
            
        Returns:
            Dict containing the response
        """
        logger.info(f"Handling document question: {query}")
        
        # Get RAG response
        rag_response = self.rag_manager.query(query, document_id)
        
        # Enhance response with conversation context
        enhanced_response = self._enhance_response_with_context(query, rag_response)
        
        return enhanced_response
    
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
                     Provide clear, concise answers based on the conversation context."""},
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
            """
            return {
                "query": query,
                "answer": help_text,
                "query_type": "system_command",
                "command": command
            }
        else:
            # Unknown command
            return {
                "query": query,
                "answer": f"Unknown command: {command}",
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
            
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You answer general questions.
                     Provide clear, concise answers based on your knowledge.
                     If the question is about a document, suggest that the user ask a document-specific question instead."""},
                    {"role": "user", "content": f"Conversation context:\n{conversation_context}\n\nQuestion: {query}"}
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
    
    def _enhance_response_with_context(self, query: str, rag_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a RAG response with conversation context.
        
        Args:
            query: User query
            rag_response: RAG response
            
        Returns:
            Dict containing the enhanced response
        """
        logger.info("Enhancing response with context")
        
        try:
            # Get conversation context
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-5:]  # Last 5 messages
            ])
            
            # Use LLM to enhance response
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """You enhance responses from a RAG system with conversation context.
                     Make the response more natural and contextual, while preserving the factual information.
                     For blind and low vision users, be clear about the location of information in the document when relevant."""},
                    {"role": "user", "content": f"Conversation context:\n{conversation_context}\n\nQuery: {query}\n\nRAG response: {rag_response['answer']}"}
                ],
                max_tokens=1000
            )
            
            enhanced_answer = response.choices[0].message.content
            
            # Return enhanced response
            return {
                "query": query,
                "answer": enhanced_answer,
                "original_answer": rag_response["answer"],
                "mode": rag_response.get("mode"),
                "is_location_query": rag_response.get("is_location_query", False),
                "location_info": rag_response.get("location_info"),
                "query_type": "document_question"
            }
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return rag_response
