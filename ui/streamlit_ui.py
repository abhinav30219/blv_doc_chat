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
from utils import logger, get_file_extension
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
        logger.info("Running Streamlit UI")
        
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
        
        # Start continuous listening if not already active
        if self.voice_enabled and not st.session_state.continuous_listening:
            self._start_continuous_voice_mode()
        
        # Main layout
        self._render_sidebar()
        self._render_main_content()
        
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
            
            logger.info("Continuous voice mode started")
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
            
            logger.info("Continuous voice mode stopped")
        except Exception as e:
            logger.error(f"Error stopping continuous voice mode: {e}")
    
    def _handle_voice_transcription(self, transcript: str):
        """
        Handle voice transcription.
        
        Args:
            transcript: Transcribed text
        """
        logger.info(f"Voice transcription: {transcript}")
        
        # Process as user input
        if transcript and st.session_state.current_document_id:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": transcript})
            
            # Process query
            try:
                response = self.agent_system.process_query(
                    transcript,
                    st.session_state.current_document_id
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
        logger.info(f"Voice command: {command_type} - {command_text}")
        
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
                key="document_uploader"
            )
            
            # Reset processed files when a new file is uploaded
            if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
                # Show process button only for files that haven't been processed
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        # Save uploaded file
                        file_path = os.path.join("temp", uploaded_file.name)
                        os.makedirs("temp", exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        try:
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
                            
                            # Announce error
                            if self.voice_enabled:
                                self.voice_interface.speak(f"Error processing document: {str(e)}")
            
            # Document selection
            if st.session_state.documents:
                st.header("Select Document")
                document_titles = [doc["title"] for doc in st.session_state.documents]
                document_ids = [doc["id"] for doc in st.session_state.documents]
                selected_index = document_titles.index(
                    self.rag_manager.get_document_metadata(st.session_state.current_document_id)["title"]
                ) if st.session_state.current_document_id in document_ids else 0
                
                selected_document = st.selectbox(
                    "Select a document",
                    document_titles,
                    index=selected_index
                )
                
                # Update current document
                selected_document_id = document_ids[document_titles.index(selected_document)]
                if selected_document_id != st.session_state.current_document_id:
                    st.session_state.current_document_id = selected_document_id
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Switched to document: {selected_document}"
                    })
            
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
            voice_enabled = st.checkbox("Enable voice input/output", value=self.voice_enabled)
            if voice_enabled != self.voice_enabled:
                self.voice_enabled = voice_enabled
                if voice_enabled:
                    self._start_continuous_voice_mode()
                    st.session_state.current_page_announced = False
                else:
                    self._stop_continuous_voice_mode()
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
            **Voice Commands:**
            - "Stop" or "Pause": Stop the current speech
            - "Continue": Continue with the previous information
            - "Search for..." or "Find...": Search the document
            - "Read document": Have the current document read aloud
            - "Read page": Have the current page/section read aloud
            - "Describe interface": Get a description of the current interface
            - "Help": Hear this help information
            
            **Text Commands:**
            - `help`: Show help
            - `clear_history`: Clear conversation history
            
            **Tips:**
            - Upload a document to start chatting
            - Ask questions about the document
            - Ask about specific locations in the document
            - Voice mode is activated by default for hands-free operation
            - You can interrupt the system by speaking at any time
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
            user_input = st.chat_input("Ask a question about the document...")
            
            # Voice controls
            st.header("Voice Controls")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if not st.session_state.continuous_listening:
                    if st.button("Start Voice Mode"):
                        self._start_continuous_voice_mode()
                        st.session_state.current_page_announced = False
                        st.rerun()
                else:
                    if st.button("Stop Voice Mode"):
                        self._stop_continuous_voice_mode()
                        st.rerun()
            
            # Manual voice recording (as backup)
            with col2:
                if not st.session_state.recording:
                    if st.button("Record Question"):
                        st.session_state.recording = True
                        self.voice_interface.start_recording()
                        st.rerun()
                else:
                    if st.button("Stop Recording"):
                        st.session_state.recording = False
                        audio_path = self.voice_interface.stop_recording()
                        st.session_state.audio_path = audio_path
                        st.rerun()
            
            # Process manual voice recording
            if st.session_state.audio_path:
                with st.spinner("Transcribing audio..."):
                    try:
                        # Run speech-to-text in a separate thread
                        loop = asyncio.new_event_loop()
                        
                        def run_async_speech_to_text():
                            try:
                                asyncio.set_event_loop(loop)
                                return loop.run_until_complete(
                                    self.voice_interface.speech_to_text(st.session_state.audio_path)
                                )
                            except Exception as e:
                                logger.error(f"Error in speech-to-text thread: {e}")
                                return f"Error transcribing audio: {str(e)}"
                        
                        thread = threading.Thread(target=run_async_speech_to_text)
                        thread.start()
                        thread.join(timeout=10)  # Add timeout to prevent hanging
                        
                        if thread.is_alive():
                            logger.warning("Speech-to-text thread timed out")
                            transcription = "Sorry, the audio transcription timed out. Please try again."
                        else:
                            # Get transcription
                            try:
                                transcription = loop.run_until_complete(
                                    self.voice_interface.speech_to_text(st.session_state.audio_path)
                                )
                            except Exception as e:
                                logger.error(f"Error getting transcription: {e}")
                                transcription = "Sorry, there was an error transcribing the audio."
                        
                        # Process the transcription
                        self._handle_voice_transcription(transcription)
                    except Exception as e:
                        logger.error(f"Error processing voice input: {e}")
                        st.error(f"Error processing voice input: {str(e)}")
                        
                        # Announce error
                        if self.voice_enabled:
                            self.voice_interface.speak(f"Error processing voice input: {str(e)}")
                    finally:
                        # Clear audio path
                        st.session_state.audio_path = None
            
            # Process user input
            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Process query
                with st.spinner("Thinking..."):
                    try:
                        response = self.agent_system.process_query(
                            user_input,
                            st.session_state.current_document_id
                        )
                        
                        # Add assistant message
                        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                        
                        # Display assistant message
                        with st.chat_message("assistant"):
                            st.markdown(response["answer"])
                        
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
