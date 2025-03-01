"""
OpenAI file manager for uploading and managing files for RAG.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY
from utils import logger, save_json, load_json

class OpenAIFileManager:
    """
    Manager for OpenAI files API integration.
    """
    
    def __init__(self, file_metadata_path: str = "./rag_data/openai_file_metadata.json"):
        """
        Initialize the OpenAI file manager.
        
        Args:
            file_metadata_path: Path to store file metadata
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.file_metadata_path = file_metadata_path
        self.file_metadata = self._load_file_metadata()
        logger.info("OpenAIFileManager initialized")
    
    def _load_file_metadata(self) -> Dict[str, Any]:
        """
        Load file metadata from disk.
        
        Returns:
            Dictionary of file metadata
        """
        try:
            if os.path.exists(self.file_metadata_path):
                return load_json(self.file_metadata_path)
            else:
                # Create empty metadata file
                metadata = {"files": {}}
                save_json(metadata, self.file_metadata_path)
                return metadata
        except Exception as e:
            logger.error(f"Error loading file metadata: {e}")
            return {"files": {}}
    
    def _save_file_metadata(self) -> None:
        """Save file metadata to disk."""
        try:
            save_json(self.file_metadata, self.file_metadata_path)
        except Exception as e:
            logger.error(f"Error saving file metadata: {e}")
    
    def upload_file(self, file_path: str, document_id: str, purpose: str = "assistants") -> str:
        """
        Upload a file to OpenAI.
        
        Args:
            file_path: Path to the file to upload
            document_id: ID of the document in our system
            purpose: Purpose of the file (assistants or fine-tune)
            
        Returns:
            OpenAI file ID
        """
        logger.info(f"Uploading file {file_path} to OpenAI")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Upload file to OpenAI
            with open(file_path, "rb") as file:
                response = self.client.files.create(
                    file=file,
                    purpose=purpose
                )
            
            # Store file metadata
            file_id = response.id
            self.file_metadata["files"][document_id] = {
                "file_id": file_id,
                "file_path": file_path,
                "purpose": purpose,
                "uploaded_at": time.time(),
                "status": "uploaded"
            }
            self._save_file_metadata()
            
            logger.info(f"File uploaded successfully with ID: {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    def get_file_id(self, document_id: str) -> Optional[str]:
        """
        Get the OpenAI file ID for a document.
        
        Args:
            document_id: ID of the document in our system
            
        Returns:
            OpenAI file ID or None if not found
        """
        if document_id in self.file_metadata["files"]:
            return self.file_metadata["files"][document_id]["file_id"]
        return None
    
    def delete_file(self, document_id: str) -> bool:
        """
        Delete a file from OpenAI.
        
        Args:
            document_id: ID of the document in our system
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Deleting file for document {document_id} from OpenAI")
        
        try:
            # Check if file exists in metadata
            if document_id not in self.file_metadata["files"]:
                logger.warning(f"No file found for document {document_id}")
                return False
            
            # Get file ID
            file_id = self.file_metadata["files"][document_id]["file_id"]
            
            # Delete file from OpenAI
            self.client.files.delete(file_id=file_id)
            
            # Update metadata
            del self.file_metadata["files"][document_id]
            self._save_file_metadata()
            
            logger.info(f"File {file_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def list_files(self) -> List[Dict[str, Any]]:
        """
        List all files uploaded to OpenAI.
        
        Returns:
            List of file metadata
        """
        try:
            # Get files from OpenAI
            response = self.client.files.list()
            
            # Format response
            files = []
            for file in response.data:
                files.append({
                    "id": file.id,
                    "filename": file.filename,
                    "bytes": file.bytes,
                    "created_at": file.created_at,
                    "purpose": file.purpose,
                    "status": file.status
                })
            
            return files
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def get_file_content(self, file_id: str) -> Optional[str]:
        """
        Get the content of a file from OpenAI.
        
        Args:
            file_id: OpenAI file ID
            
        Returns:
            File content as string or None if error
        """
        try:
            # Get file content from OpenAI
            response = self.client.files.content(file_id=file_id)
            
            # Return content as string
            return response.decode("utf-8")
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return None
    
    def create_assistant_with_file(self, 
                                  document_id: str, 
                                  name: str = "Document Assistant",
                                  instructions: str = "You are a helpful assistant that answers questions about documents.",
                                  model: str = "gpt-4o") -> Optional[str]:
        """
        Create an OpenAI assistant with a file.
        
        Args:
            document_id: ID of the document in our system
            name: Name of the assistant
            instructions: Instructions for the assistant
            model: Model to use
            
        Returns:
            Assistant ID or None if error
        """
        try:
            # Get file ID
            file_id = self.get_file_id(document_id)
            if not file_id:
                logger.warning(f"No file found for document {document_id}")
                return None
            
            # Create assistant
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=model,
                tools=[{"type": "retrieval"}],
                file_ids=[file_id]
            )
            
            # Update metadata
            if document_id in self.file_metadata["files"]:
                self.file_metadata["files"][document_id]["assistant_id"] = assistant.id
                self._save_file_metadata()
            
            logger.info(f"Assistant created with ID: {assistant.id}")
            return assistant.id
        except Exception as e:
            logger.error(f"Error creating assistant: {e}")
            return None
    
    def get_assistant_id(self, document_id: str) -> Optional[str]:
        """
        Get the OpenAI assistant ID for a document.
        
        Args:
            document_id: ID of the document in our system
            
        Returns:
            OpenAI assistant ID or None if not found
        """
        if document_id in self.file_metadata["files"] and "assistant_id" in self.file_metadata["files"][document_id]:
            return self.file_metadata["files"][document_id]["assistant_id"]
        return None
    
    def delete_assistant(self, document_id: str) -> bool:
        """
        Delete an OpenAI assistant.
        
        Args:
            document_id: ID of the document in our system
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get assistant ID
            assistant_id = self.get_assistant_id(document_id)
            if not assistant_id:
                logger.warning(f"No assistant found for document {document_id}")
                return False
            
            # Delete assistant
            self.client.beta.assistants.delete(assistant_id=assistant_id)
            
            # Update metadata
            if document_id in self.file_metadata["files"] and "assistant_id" in self.file_metadata["files"][document_id]:
                del self.file_metadata["files"][document_id]["assistant_id"]
                self._save_file_metadata()
            
            logger.info(f"Assistant {assistant_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting assistant: {e}")
            return False
    
    def create_thread(self) -> Optional[str]:
        """
        Create an OpenAI thread for conversation.
        
        Returns:
            Thread ID or None if error
        """
        try:
            # Create thread
            thread = self.client.beta.threads.create()
            
            logger.info(f"Thread created with ID: {thread.id}")
            return thread.id
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            return None
    
    def add_message_to_thread(self, thread_id: str, content: str, role: str = "user") -> Optional[str]:
        """
        Add a message to an OpenAI thread.
        
        Args:
            thread_id: Thread ID
            content: Message content
            role: Message role (user or assistant)
            
        Returns:
            Message ID or None if error
        """
        try:
            # Add message to thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content
            )
            
            logger.info(f"Message added to thread {thread_id}")
            return message.id
        except Exception as e:
            logger.error(f"Error adding message to thread: {e}")
            return None
    
    def run_assistant(self, 
                     thread_id: str, 
                     assistant_id: str, 
                     instructions: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Run an OpenAI assistant on a thread.
        
        Args:
            thread_id: Thread ID
            assistant_id: Assistant ID
            instructions: Optional additional instructions
            
        Returns:
            Run result or None if error
        """
        try:
            # Create run
            run_params = {
                "thread_id": thread_id,
                "assistant_id": assistant_id
            }
            
            if instructions:
                run_params["instructions"] = instructions
            
            run = self.client.beta.threads.runs.create(**run_params)
            
            # Wait for run to complete
            while run.status in ["queued", "in_progress"]:
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
            
            # Get messages
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id
            )
            
            # Format response
            response = {
                "run_id": run.id,
                "status": run.status,
                "messages": []
            }
            
            for message in messages.data:
                content_parts = []
                for content in message.content:
                    if content.type == "text":
                        content_parts.append(content.text.value)
                
                response["messages"].append({
                    "role": message.role,
                    "content": "\n".join(content_parts)
                })
            
            logger.info(f"Assistant run completed with status: {run.status}")
            return response
        except Exception as e:
            logger.error(f"Error running assistant: {e}")
            return None
    
    def query_document(self, document_id: str, query: str) -> Optional[str]:
        """
        Query a document using OpenAI assistant.
        
        Args:
            document_id: ID of the document in our system
            query: Query to ask
            
        Returns:
            Response text or None if error
        """
        try:
            # Get assistant ID or create one
            assistant_id = self.get_assistant_id(document_id)
            if not assistant_id:
                assistant_id = self.create_assistant_with_file(document_id)
                if not assistant_id:
                    return None
            
            # Create thread
            thread_id = self.create_thread()
            if not thread_id:
                return None
            
            # Add message to thread
            message_id = self.add_message_to_thread(thread_id, query)
            if not message_id:
                return None
            
            # Run assistant
            result = self.run_assistant(thread_id, assistant_id)
            if not result or result["status"] != "completed":
                return None
            
            # Get assistant response
            for message in result["messages"]:
                if message["role"] == "assistant":
                    return message["content"]
            
            return None
        except Exception as e:
            logger.error(f"Error querying document: {e}")
            return None
