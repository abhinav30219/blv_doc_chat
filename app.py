import os
import streamlit as st
from dotenv import load_dotenv
import threading
import time

# Import modules
from document_processor.processor import DocumentProcessor
from rag_system.lightrag_manager import LightRAGManager
from voice_interface.openai_voice import OpenAIVoiceInterface
from multi_agent.agent_system import MultiAgentSystem
from ui.streamlit_ui import StreamlitUI
from utils import logger
from config import VOICE_ENABLED_BY_DEFAULT, VOICE_STYLE

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="BLV Document Chat",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    document_processor = DocumentProcessor()
    rag_manager = LightRAGManager()
    voice_interface = OpenAIVoiceInterface()
    
    # Set voice properties
    voice_interface.set_voice_properties(style=VOICE_STYLE)
    
    agent_system = MultiAgentSystem(rag_manager)
    
    # Initialize UI with voice mode enabled by default
    ui = StreamlitUI(document_processor, rag_manager, voice_interface, agent_system)
    
    # Add welcome message for screen readers
    if "welcome_announced" not in st.session_state:
        st.session_state.welcome_announced = False
        
        # Start a thread to announce welcome message after UI is loaded
        def announce_welcome():
            # Wait for UI to load
            time.sleep(2)
            
            # Announce welcome message
            welcome_message = """
            Welcome to Document Chat for Blind and Low Vision Users. 
            This application allows you to chat with your documents using voice or text.
            Voice mode is activated by default. You can speak to interact with the application.
            Say "help" at any time to get assistance.
            """
            voice_interface.speak(welcome_message)
            st.session_state.welcome_announced = True
        
        threading.Thread(target=announce_welcome, daemon=True).start()
    
    # Run the UI
    ui.run()

if __name__ == "__main__":
    main()
