#!/usr/bin/env python3
"""
Demo script for the BLV Document Chat application.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor.processor import DocumentProcessor
from rag_system.lightrag_manager import LightRAGManager
from voice_interface.hume_voice import HumeVoiceInterface
from multi_agent.agent_system import MultiAgentSystem
from utils import logger, create_directories

def main():
    """Run a demo of the BLV Document Chat application."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Demo of BLV Document Chat")
    parser.add_argument("document", help="Path to document file")
    parser.add_argument("--query", help="Query to ask about the document")
    parser.add_argument("--voice", action="store_true", help="Use voice output")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create necessary directories
    create_directories()
    
    # Initialize components
    print("Initializing components...")
    document_processor = DocumentProcessor()
    rag_manager = LightRAGManager()
    voice_interface = HumeVoiceInterface()
    agent_system = MultiAgentSystem(rag_manager)
    
    # Process document
    print(f"Processing document: {args.document}")
    document_content = document_processor.process_document(args.document)
    document_id = rag_manager.add_document(document_content)
    print(f"Document processed: {document_content['metadata']['title']}")
    
    # Print document info
    print("\nDocument Info:")
    print(f"Title: {document_content['metadata']['title']}")
    print(f"Pages: {document_content['metadata']['num_pages']}")
    print(f"Sections: {len(document_content['metadata']['sections'])}")
    print(f"Images: {len(document_content['metadata']['images'])}")
    
    # Process query if provided
    if args.query:
        print(f"\nProcessing query: {args.query}")
        response = agent_system.process_query(args.query, document_id)
        
        print("\nResponse:")
        print(response["answer"])
        
        # Use voice output if requested
        if args.voice:
            print("\nGenerating voice output...")
            audio_path = voice_interface.text_to_speech(response["answer"])
            print(f"Voice output saved to: {audio_path}")
            print("Playing audio...")
            voice_interface.play_audio(audio_path)
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter query: ")
            if query.lower() == "exit":
                break
            
            response = agent_system.process_query(query, document_id)
            
            print("\nResponse:")
            print(response["answer"])
            
            # Use voice output if requested
            if args.voice:
                print("\nGenerating voice output...")
                audio_path = voice_interface.text_to_speech(response["answer"])
                print(f"Voice output saved to: {audio_path}")
                print("Playing audio...")
                voice_interface.play_audio(audio_path)
    
    print("\nDemo complete.")

if __name__ == "__main__":
    main()
