"""
Streamlit UI for the BLV Document Chat application.
"""

import os
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union
import streamlit as st
import pandas as pd

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    UI_TITLE,
    UI_DESCRIPTION,
    UI_THEME,
    SUPPORTED_DOCUMENT_TYPES,
    VOICE_ENABLED_BY_DEFAULT,
    VOICE_CONTINUOUS_GUIDANCE,
    VOICE_SPEED,
    VOICE_VOLUME,
    VOICE_INTERRUPTION_ENABLED
)
from utils import logger, pipeline_logger, PipelineTimer, get_file_extension
from document_processor.processor import DocumentProcessor
from rag_system.lightrag_manager import LightRAGManager
from voice_interface.openai_voice import OpenAIVoiceInterface
from multi_agent.agent_system import MultiAgentSystem

class StreamlitUI:
    """
    Streamlit UI for the BLV Document Chat application.
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        rag_manager: LightRAGManager,
        voice_interface: OpenAIVoiceInterface,
        agent_system: MultiAgentSystem
    ):
        """
        Initialize the Streamlit UI.
        
        Args:
            document_processor: Document processor
            rag_manager: LightRAG manager
            voice_interface: OpenAI voice interface
            agent_system: Multi-agent system
        """
        self.document_processor = document_processor
        self.rag_manager = rag_manager
        self.voice_interface = voice_interface
        self.agent_system = agent_system
        self.voice_enabled = VOICE_ENABLED_BY_DEFAULT
        self.voice_speed = VOICE_SPEED
        self.voice_volume = VOICE_VOLUME
        self.continuous_guidance = VOICE_CONTINUOUS_GUIDANCE
        self.voice_interruption = VOICE_INTERRUPTION_ENABLED
        self.current_page_elements = []
        logger.info("StreamlitUI initialized")
    
    def run(self):
        """Run the Streamlit UI."""
        if pipeline_logger:
            pipeline_logger.info("Starting Streamlit UI")
        
        # Page config is set in app.py
        
        # Set theme
        if UI_THEME == "dark":
            st.markdown("""
            <style>
                body {
                    color: white;
                    background-color: #0e1117;
                }
            </style>
            """, unsafe_allow_html=True)
        
        # Initialize session state
        with PipelineTimer("Session State Initialization"):
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "current_document_id" not in st.session_state:
                st.session_state.current_document_id = None
            if "documents" not in st.session_state:
                st.session_state.documents = []
            if "recording" not in st.session_state:
                st.session_state.recording = False
            if "audio_path" not in st.session_state:
                st.session_state.audio_path = None
            if "voice_mode_active" not in st.session_state:
                st.session_state.voice_mode_active = self.voice_enabled
            if "continuous_listening" not in st.session_state:
                st.session_state.continuous_listening = False
            if "current_page_announced" not in st.session_state:
                st.session_state.current_page_announced = False
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = set()
            # Processing state flags
            if "processing_document" not in st.session_state:
                st.session_state.processing_document = False
            if "processing_query" not in st.session_state:
                st.session_state.processing_query = False
            if "processing_start_time" not in st.session_state:
                st.session_state.processing_start_time = 0
            if "announcement_made" not in st.session_state:
                st.session_state.announcement_made = False
            if "follow_up_announcement_made" not in st.session_state:
                st.session_state.follow_up_announcement_made = False
            
            if pipeline_logger:
                pipeline_logger.info("Session state initialized")
        
        # Continuous listening disabled for stability
        # if self.voice_enabled and not st.session_state.continuous_listening:
        #     self._start_continuous_voice_mode()
        
        # Main layout
        with PipelineTimer("UI Rendering"):
            self._render_sidebar()
            self._render_main_content()
            
            if pipeline_logger:
                pipeline_logger.info("UI rendered")
        
        # Announce page content if voice mode is active
        if self.voice_enabled and not st.session_state.current_page_announced:
            self._announce_current_page()
            st.session_state.current_page_announced = True
    
    def _start_continuous_voice_mode(self):
        """Start continuous voice mode."""
        try:
            # Set session state
            st.session_state.continuous_listening = True
            st.session_state.current_page_announced = False
            
            # Start continuous listening
            self.voice_interface.start_continuous_listening(
                transcription_callback=self._handle_voice_transcription,
                command_callback=self._handle_voice_command
            )
        except Exception as e:
            logger.error(f"Error starting continuous voice mode: {e}")
            st.error(f"Error starting voice mode: {str(e)}")
    
    def _stop_continuous_voice_mode(self):
        """Stop continuous voice mode."""
        try:
            # Stop continuous listening
            self.voice_interface.stop_continuous_listening()
            
            # Set session state
            st.session_state.continuous_listening = False
        except Exception as e:
            logger.error(f"Error stopping continuous voice mode: {e}")
    
    def _handle_voice_transcription(self, transcript: str):
        """
        Handle voice transcription.
        
        Args:
            transcript: Transcribed text
        """
        # Process as user input
        if transcript and st.session_state.current_document_id:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": transcript})
            
            # Process query
            try:
                # Use selected document IDs for query
                if "selected_document_ids" in st.session_state and st.session_state.selected_document_ids:
                    document_ids = st.session_state.selected_document_ids
                    if pipeline_logger:
                        pipeline_logger.info(f"Voice query across {len(document_ids)} selected documents")
                else:
                    # Fallback to current document ID for backward compatibility
                    document_ids = [st.session_state.current_document_id] if st.session_state.current_document_id else None
                    if pipeline_logger:
                        pipeline_logger.info(f"Voice query for single document: {st.session_state.current_document_id}")
                
                response = self.agent_system.process_query(
                    transcript,
                    document_ids
                )
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                
                # Speak the response
                self.voice_interface.speak(response["answer"])
                
                # Force a rerun to update the UI
                st.rerun()
            except Exception as e:
                logger.error(f"Error processing voice query: {e}")
                error_message = f"I'm sorry, I encountered an error while processing your query: {str(e)}"
                
                # Add error message
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
                # Speak the error message
                self.voice_interface.speak(error_message)
                
                # Force a rerun to update the UI
                st.rerun()
    
    def _handle_voice_command(self, command_type: str, command_text: str):
        """
        Handle voice command.
        
        Args:
            command_type: Type of command
            command_text: Command text
        """
        if command_type == "search" and st.session_state.current_document_id:
            # Process as search query
            self._handle_voice_transcription(command_text)
        elif command_type == "read_document" and st.session_state.current_document_id:
            # Read the current document
            document_metadata = self.rag_manager.get_document_metadata(st.session_state.current_document_id)
            if document_metadata:
                document_content = f"Reading document {document_metadata['title']}. "
                
                # Add sections if available
                if document_metadata.get("sections"):
                    sections_text = []
                    for section in document_metadata["sections"]:
                        if section.get("title") and section.get("content"):
                            sections_text.append(f"{section['title']}: {section['content']}")
                    
                    if sections_text:
                        document_content += "The document contains the following sections: " + ". ".join(sections_text)
                
                self.voice_interface.speak(document_content)
        elif command_type == "read_page" and st.session_state.current_document_id:
            # Read the current page/section
            document_metadata = self.rag_manager.get_document_metadata(st.session_state.current_document_id)
            if document_metadata and document_metadata.get("sections"):
                # Just read the first section as an example
                section = document_metadata["sections"][0]
                if section.get("title") and section.get("content"):
                    self.voice_interface.speak(f"{section['title']}: {section['content']}")
        elif command_type == "describe_interface":
            # Describe the current interface
            self._announce_current_page()
    
    def _announce_current_page(self):
        """Announce the current page content."""
        if not self.voice_enabled:
            return
        
        # Check if we're processing a document or query
        if st.session_state.processing_document:
            # Document processing announcements
            current_time = time.time()
            if not st.session_state.announcement_made:
                # Initial announcement
                self.voice_interface.speak("Please wait while I learn from the document you uploaded.")
                st.session_state.announcement_made = True
                st.session_state.processing_start_time = current_time
            elif not st.session_state.follow_up_announcement_made and (current_time - st.session_state.processing_start_time) > 10:
                # Follow-up announcement after 10 seconds
                self.voice_interface.speak("Still analyzing your document. This may take a moment.")
                st.session_state.follow_up_announcement_made = True
            return
            
        elif st.session_state.processing_query:
            # Query processing announcements
            current_time = time.time()
            if not st.session_state.announcement_made:
                # Initial announcement
                self.voice_interface.speak("Please wait while I think about your question.")
                st.session_state.announcement_made = True
                st.session_state.processing_start_time = current_time
            elif not st.session_state.follow_up_announcement_made and (current_time - st.session_state.processing_start_time) > 5:
                # Follow-up announcement after 5 seconds
                self.voice_interface.speak("Almost done generating your answer.")
                st.session_state.follow_up_announcement_made = True
            return
        
        # Regular page announcement (only when not processing)
        # Collect page elements
        elements = []
        
        # Add page title
        elements.append(("header", "Document Chat for BLV Users"))
        
        # Add document status
        if st.session_state.current_document_id:
            document_metadata = self.rag_manager.get_document_metadata(st.session_state.current_document_id)
            if document_metadata:
                elements.append(("text", f"Current document: {document_metadata['title']}"))
                elements.append(("text", f"This document has {document_metadata['num_pages']} pages and {len(document_metadata['sections'])} sections."))
        else:
            elements.append(("text", "No document loaded. Please upload a document to start chatting."))
        
        # Add chat messages
        if st.session_state.messages:
            elements.append(("header", "Conversation"))
            for message in st.session_state.messages[-3:]:  # Last 3 messages
                elements.append((message["role"], message["content"]))
        
        # Announce elements
        self.voice_interface.announce_page_content(elements)
    
    def _render_sidebar(self):
        """Render the sidebar."""
        with st.sidebar:
            st.title("📄 Document Chat")
            st.markdown(UI_DESCRIPTION)
            
            # Document upload
            st.header("Upload Document")
            uploaded_file = st.file_uploader(
                "Upload a document",
                type=[ext[1:] for ext in SUPPORTED_DOCUMENT_TYPES],
                key="document_uploader",
                help="Select and upload a pdf document to start interacting"
            )
            
            # Reset processed files when a new file is uploaded
            if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
                # Show process button only for files that haven't been processed
                if st.button("Process Document", help="Click to process the uploaded document for text analysis."):
                    # Set processing document flag
                    st.session_state.processing_document = True
                    st.session_state.announcement_made = False
                    st.session_state.follow_up_announcement_made = False
                    
                    with st.spinner("Processing document..."):
                        if pipeline_logger:
                            pipeline_logger.info(f"Starting document processing: {uploaded_file.name}")
                        
                        # Save uploaded file
                        file_path = os.path.join("temp", uploaded_file.name)
                        os.makedirs("temp", exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        try:
                            with PipelineTimer("Document Processing"):
                                document_content = self.document_processor.process_document(file_path)
                                document_id = self.rag_manager.add_document(document_content)
                                
                                # Update session state
                                st.session_state.current_document_id = document_id
                                st.session_state.documents = self.rag_manager.list_documents()
                                
                                # Mark this file as processed
                                st.session_state.processed_files.add(uploaded_file.name)
                                
                                # Add system message
                                st.session_state.messages.append({
                                    "role": "system",
                                    "content": f"Document '{document_content['metadata']['title']}' processed successfully."
                                })
                                
                                if pipeline_logger:
                                    pipeline_logger.info(f"Document processed successfully: {document_content['metadata']['title']}")
                            
                            # Reset processing flags
                            st.session_state.processing_document = False
                            
                            # Announce document processed
                            if self.voice_enabled:
                                self.voice_interface.speak(
                                    f"Document {document_content['metadata']['title']} processed successfully. "
                                    f"It has {document_content['metadata']['num_pages']} pages and "
                                    f"{len(document_content['metadata']['sections'])} sections."
                                )
                            
                            st.success(f"Document processed: {document_content['metadata']['title']}")
                            
                            # Force a rerun to update the UI
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
                            logger.error(f"Error processing document: {e}")
                            
                            if pipeline_logger:
                                pipeline_logger.info(f"Document processing failed: {str(e)}")
                            
                            # Announce error
                            if self.voice_enabled:
                                self.voice_interface.speak(f"Error processing document: {str(e)}")
            
            # Document selection
            if st.session_state.documents:
                st.header("Document Selection")
                
                # Initialize selected_document_ids in session state if not present
                if "selected_document_ids" not in st.session_state:
                    # Default to all documents selected
                    st.session_state.selected_document_ids = [doc["id"] for doc in st.session_state.documents]
                
                # Add "Select All" and "Deselect All" buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select All", help="Select all available documents for querying."):
                        st.session_state.selected_document_ids = [doc["id"] for doc in st.session_state.documents]
                        st.rerun()
                with col2:
                    if st.button("Deselect All", help="Deselect all documents to exclude them from queries."):
                        st.session_state.selected_document_ids = []
                        st.rerun()
                
                # Create checkboxes for each document
                st.write("Select documents to include in search:")
                for doc in st.session_state.documents:
                    doc_selected = st.checkbox(
                        doc["title"],
                        value=doc["id"] in st.session_state.selected_document_ids,
                        key=f"doc_checkbox_{doc['id']}",
                        help=f"Check to focus on the document '{doc['title']}' in your search."
                    )
                    
                    # Update selected_document_ids based on checkbox state
                    if doc_selected and doc["id"] not in st.session_state.selected_document_ids:
                        st.session_state.selected_document_ids.append(doc["id"])
                    elif not doc_selected and doc["id"] in st.session_state.selected_document_ids:
                        st.session_state.selected_document_ids.remove(doc["id"])
                
                # Set current document for backward compatibility
                if st.session_state.selected_document_ids:
                    if st.session_state.current_document_id not in st.session_state.selected_document_ids:
                        st.session_state.current_document_id = st.session_state.selected_document_ids[0]
                        
                        # Get document title
                        current_doc_title = "Unknown document"
                        for doc in st.session_state.documents:
                            if doc["id"] == st.session_state.current_document_id:
                                current_doc_title = doc["title"]
                                break
                        
                        if pipeline_logger:
                            pipeline_logger.info(f"Set current document to: {current_doc_title}")
                else:
                    st.session_state.current_document_id = None
                    if pipeline_logger:
                        pipeline_logger.info("No documents selected")
            
            # Document info
            if st.session_state.current_document_id:
                st.header("Document Info")
                document_metadata = self.rag_manager.get_document_metadata(st.session_state.current_document_id)
                if document_metadata:
                    st.write(f"Title: {document_metadata['title']}")
                    st.write(f"Pages: {document_metadata['num_pages']}")
                    st.write(f"Sections: {len(document_metadata['sections'])}")
                    st.write(f"Images: {len(document_metadata['images'])}")
            
            # Accessibility options
            st.header("Accessibility Options")
            voice_enabled = st.checkbox("Enable voice output", value=self.voice_enabled)
            if voice_enabled != self.voice_enabled:
                self.voice_enabled = voice_enabled
                # Voice input disabled for stability
                # if voice_enabled:
                #     self._start_continuous_voice_mode()
                #     st.session_state.current_page_announced = False
                # else:
                #     self._stop_continuous_voice_mode()
                st.rerun()
            
            if voice_enabled:
                voice_speed = st.slider("Voice speed", min_value=0.5, max_value=2.0, value=self.voice_speed, step=0.1)
                if voice_speed != self.voice_speed:
                    self.voice_speed = voice_speed
                    self.voice_interface.set_voice_properties(speed=voice_speed)
                
                voice_volume = st.slider("Voice volume", min_value=0.1, max_value=1.0, value=self.voice_volume, step=0.1)
                if voice_volume != self.voice_volume:
                    self.voice_volume = voice_volume
                    self.voice_interface.set_voice_properties(volume=voice_volume)
            
            # Help
            st.header("Help")
            st.markdown("""
            **Text Commands:**
            - `help`: Show help
            - `clear_history`: Clear conversation history
            
            **Tips:**
            - Upload a document to start chatting
            - Ask questions about the document
            - Ask about specific locations in the document
            - Enable voice output to have responses read aloud
            """)
    
    def _render_main_content(self):
        """Render the main content."""
        st.header("Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if st.session_state.current_document_id:
            # Text input
            user_input = st.chat_input(
                "Ask a question about the document...",
                help="Type a question about the document and press Enter to submit."
            )
            
            # Voice recording disabled for stability
            st.info("Voice recording has been disabled for stability. Please use the text input above to ask questions.")
            
            # Process user input
            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Set processing query flag
                st.session_state.processing_query = True
                st.session_state.announcement_made = False
                st.session_state.follow_up_announcement_made = False
                
                if pipeline_logger:
                    pipeline_logger.info(f"Processing query: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'")
                
                # Process query
                with st.spinner("Thinking..."):
                    try:
                        with PipelineTimer("Query Processing"):
                            # Use selected document IDs for query
                            if "selected_document_ids" in st.session_state and st.session_state.selected_document_ids:
                                document_ids = st.session_state.selected_document_ids
                                if pipeline_logger:
                                    pipeline_logger.info(f"Querying across {len(document_ids)} selected documents")
                            else:
                                # Fallback to current document ID for backward compatibility
                                document_ids = [st.session_state.current_document_id] if st.session_state.current_document_id else None
                                if pipeline_logger:
                                    pipeline_logger.info(f"Querying single document: {st.session_state.current_document_id}")
                            
                            response = self.agent_system.process_query(
                                user_input,
                                document_ids
                            )
                            
                            # Reset processing flags
                            st.session_state.processing_query = False
                            
                            # Add assistant message
                            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                            
                            # Display assistant message
                            with st.chat_message("assistant"):
                                st.markdown(response["answer"])
                            
                            if pipeline_logger:
                                pipeline_logger.info(f"Query processed successfully ({len(response['answer'])} chars)")
                        
                        # Automatically speak the response if voice mode is enabled
                        if self.voice_enabled:
                            self.voice_interface.speak(response["answer"])
                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        error_message = f"I'm sorry, I encountered an error while processing your query: {str(e)}"
                        
                        # Add error message
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        
                        # Display error message
                        with st.chat_message("assistant"):
                            st.markdown(error_message)
                        
                        if pipeline_logger:
                            pipeline_logger.info(f"Query processing failed: {str(e)}")
                        
                        # Speak error message if voice mode is enabled
                        if self.voice_enabled:
                            self.voice_interface.speak(error_message)
                        
                        # Set response for text-to-speech
                        response = {"answer": error_message}
                
                # Text-to-speech controls - automatically checked when voice mode is enabled
                st.checkbox("Read response aloud", value=self.voice_enabled, key="read_aloud", 
                           help="When checked, responses will be read aloud automatically")
                
                # If read aloud is checked and we have a response, generate speech
                if st.session_state.read_aloud and "messages" in st.session_state and st.session_state.messages:
                    last_message = st.session_state.messages[-1]
                    if last_message["role"] == "assistant":
                        with st.spinner("Generating speech..."):
                            try:
                                audio_path = self.voice_interface.text_to_speech(last_message["content"])
                                st.audio(audio_path)
                            except Exception as e:
                                logger.error(f"Error generating speech: {e}")
                                st.error(f"Error generating speech: {str(e)}")
        else:
            st.info("Please upload a document to start chatting.")
