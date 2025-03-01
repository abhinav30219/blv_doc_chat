"""
Configuration settings for the BLV Document Chat application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUME_API_KEY = os.getenv("HUME_API_KEY")

# LLM Settings
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"

# RAG Settings
RAG_WORKING_DIR = "./rag_data"
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100

# Document Processing Settings
SUPPORTED_DOCUMENT_TYPES = [".pdf", ".docx", ".doc", ".txt", ".pptx"]
MAX_IMAGE_SIZE = (1000, 1000)  # Maximum width and height for images
OCR_LANGUAGES = ["eng"]  # Languages for OCR

# Voice Interface Settings
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "mp3"
VOICE_ENABLED_BY_DEFAULT = True
VOICE_CONTINUOUS_GUIDANCE = True
VOICE_SPEED = 1.0
VOICE_VOLUME = 0.8
VOICE_INTERRUPTION_ENABLED = True
VOICE_STYLE = "sage"  # OpenAI voice style: alloy, echo, fable, onyx, nova, shimmer, or sage

# OpenAI API Settings
OPENAI_TTS_MODEL = "tts-1-hd"  # or "tts-1" for lower quality
OPENAI_STT_MODEL = "whisper-1"
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_AUDIO_MODEL = "gpt-4o-audio-preview"  # Audio model for direct audio interactions
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
USE_AUDIO_MODEL = True  # Whether to use the audio model for voice interactions

# OpenAI File API Settings
OPENAI_FILE_ENABLED = True  # Whether to use OpenAI Files API
OPENAI_FILE_METADATA_PATH = "./rag_data/openai_file_metadata.json"
OPENAI_ASSISTANT_MODEL = "gpt-4o"

# UI Settings
UI_THEME = "light"
UI_TITLE = "Document Chat for BLV Users"
UI_DESCRIPTION = "Chat with your documents using voice or text"

# Logging Settings
LOG_LEVEL = "DEBUG"
LOG_FILE = "app.log"

# Paths
TEMP_DIR = "./temp"
OUTPUT_DIR = "./output"
