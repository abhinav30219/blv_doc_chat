#!/usr/bin/env python3
"""
Test script for the BLV Document Chat application.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor.processor import DocumentProcessor
from rag_system.lightrag_manager import LightRAGManager
from voice_interface.hume_voice import HumeVoiceInterface
from multi_agent.agent_system import MultiAgentSystem

class TestDocumentProcessor(unittest.TestCase):
    """Test the DocumentProcessor class."""
    
    @patch('document_processor.processor.openai.OpenAI')
    def test_init(self, mock_openai):
        """Test initialization."""
        processor = DocumentProcessor()
        self.assertIsNotNone(processor)
    
    @patch('document_processor.processor.openai.OpenAI')
    @patch('document_processor.processor.partition_pdf')
    @patch('document_processor.processor.pypdf.PdfReader')
    def test_process_pdf(self, mock_pdf_reader, mock_partition_pdf, mock_openai):
        """Test processing a PDF document."""
        # Mock the necessary components
        mock_pdf_reader.return_value.pages = [MagicMock(), MagicMock()]
        mock_partition_pdf.return_value = []
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            # Create processor
            processor = DocumentProcessor()
            
            # Mock the extract_document_structure method
            processor._extract_document_structure = MagicMock(return_value=[])
            processor._extract_images_from_pdf = MagicMock(return_value=[])
            
            # Process the document
            result = processor._process_pdf(temp_file.name)
            
            # Check the result
            self.assertIsInstance(result, dict)
            self.assertIn('metadata', result)
            self.assertIn('elements', result)
            self.assertIn('images', result)

class TestLightRAGManager(unittest.TestCase):
    """Test the LightRAGManager class."""
    
    @patch('rag_system.lightrag_manager.LightRAG')
    @patch('rag_system.lightrag_manager.openai.OpenAI')
    def test_init(self, mock_openai, mock_lightrag):
        """Test initialization."""
        manager = LightRAGManager()
        self.assertIsNotNone(manager)
    
    @patch('rag_system.lightrag_manager.LightRAG')
    @patch('rag_system.lightrag_manager.openai.OpenAI')
    def test_query(self, mock_openai, mock_lightrag):
        """Test querying the RAG system."""
        # Mock the necessary components
        mock_lightrag.return_value.query.return_value = "Test response"
        
        # Create manager
        manager = LightRAGManager()
        
        # Mock the analyze_query_for_location method
        manager._analyze_query_for_location = MagicMock(return_value=(False, None))
        
        # Query the RAG system
        result = manager.query("Test query")
        
        # Check the result
        self.assertIsInstance(result, dict)
        self.assertIn('query', result)
        self.assertIn('answer', result)
        self.assertIn('mode', result)
        self.assertIn('is_location_query', result)

class TestHumeVoiceInterface(unittest.TestCase):
    """Test the HumeVoiceInterface class."""
    
    @patch('voice_interface.hume_voice.HumeClient')
    @patch('voice_interface.hume_voice.AsyncHumeClient')
    def test_init(self, mock_async_hume, mock_hume):
        """Test initialization."""
        interface = HumeVoiceInterface()
        self.assertIsNotNone(interface)
    
    @patch('voice_interface.hume_voice.HumeClient')
    @patch('voice_interface.hume_voice.AsyncHumeClient')
    @patch('voice_interface.hume_voice.base64')
    def test_text_to_speech(self, mock_base64, mock_async_hume, mock_hume):
        """Test text-to-speech conversion."""
        # Mock the necessary components
        mock_hume.return_value.tts.synthesize_json.return_value.audio = "test_audio"
        mock_base64.b64decode.return_value = b"test_audio_decoded"
        
        # Create interface
        interface = HumeVoiceInterface()
        
        # Mock the open function
        with patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            # Convert text to speech
            result = interface.text_to_speech("Test text")
            
            # Check the result
            self.assertIsInstance(result, str)

class TestMultiAgentSystem(unittest.TestCase):
    """Test the MultiAgentSystem class."""
    
    @patch('multi_agent.agent_system.openai.OpenAI')
    def test_init(self, mock_openai):
        """Test initialization."""
        rag_manager = MagicMock()
        agent_system = MultiAgentSystem(rag_manager)
        self.assertIsNotNone(agent_system)
    
    @patch('multi_agent.agent_system.openai.OpenAI')
    def test_process_query(self, mock_openai):
        """Test processing a query."""
        # Mock the necessary components
        rag_manager = MagicMock()
        
        # Create agent system
        agent_system = MultiAgentSystem(rag_manager)
        
        # Mock the analyze_query method
        agent_system._analyze_query = MagicMock(return_value={"query_type": "document_question"})
        
        # Mock the handle_document_question method
        agent_system._handle_document_question = MagicMock(return_value={
            "query": "Test query",
            "answer": "Test answer",
            "query_type": "document_question"
        })
        
        # Process a query
        result = agent_system.process_query("Test query")
        
        # Check the result
        self.assertIsInstance(result, dict)
        self.assertIn('query', result)
        self.assertIn('answer', result)
        self.assertIn('query_type', result)

if __name__ == '__main__':
    unittest.main()
