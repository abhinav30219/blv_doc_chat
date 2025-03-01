# BLV Document Chat

A multimodal LLM and RAG-powered document chat application designed specifically for Blind and Low Vision (BLV) users.

![BLV Document Chat](https://via.placeholder.com/800x400?text=BLV+Document+Chat)

## Features

- **Document Processing**: Extract text, formatting details, images, figures, and charts from various document types (PDF, DOCX, TXT)
- **RAG System**: Store and retrieve document content using LightRAG with OpenAI Files API integration
- **Voice Interface**: Natural speech-to-text and text-to-speech using OpenAI's audio APIs, including GPT-4o audio preview
- **Multi-Agent System**: Coordinate document analysis and question answering
- **Accessible UI**: Streamlit interface designed for BLV users with continuous voice guidance

## Requirements

- Python 3.9+
- OpenAI API key (with access to GPT-4o, Whisper, and TTS models)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blv_doc_chat.git
cd blv_doc_chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the template:
```bash
cp .env.template .env
```

4. Edit the `.env` file to add your API keys and configure settings:
```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUME_API_KEY=your_hume_api_key_here  # Optional, for Hume voice interface

# LLM Settings
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large

# Voice Interface Settings
VOICE_STYLE=sage  # OpenAI voice style: alloy, echo, fable, onyx, nova, shimmer, or sage
VOICE_SPEED=1.0
VOICE_VOLUME=0.8
VOICE_ENABLED_BY_DEFAULT=true

# OpenAI API Settings
OPENAI_TTS_MODEL=tts-1-hd  # or tts-1 for lower quality
OPENAI_STT_MODEL=whisper-1
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_AUDIO_MODEL=gpt-4o-audio-preview  # Audio model for direct audio interactions
```

> **IMPORTANT**: Never commit your `.env` file to version control. It contains sensitive API keys.

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Upload a document (PDF, DOCX, TXT)

3. Ask questions about the document using text or voice input

4. Receive answers with location information and context

## Query Types

The application supports different types of queries:

- **Content Queries**: "What does the document say about design principles?"
- **Location Queries**: "What's on page 5?" or "What's in the third section?"
- **Conversation Queries**: "What did we talk about earlier?"
- **System Commands**: "help", "clear_history"

## Architecture

The application is built with a modular architecture:

- **Document Processor**: Extracts text, images, and structure from documents
- **RAG System**: Stores and retrieves document content using LightRAG with OpenAI Files API integration
- **Voice Interface**: Handles speech-to-text and text-to-speech using OpenAI's audio APIs
- **Multi-Agent System**: Coordinates document analysis and question answering
- **UI**: Provides a Streamlit interface for the application

## Accessibility Features

- **Voice Input/Output**: Hands-free operation using OpenAI's natural-sounding voices
- **GPT-4o Audio Preview**: Direct audio-to-audio conversations for more natural interactions
- **Continuous Voice Guidance**: System continuously guides users through voice about what's on screen
- **Natural Language Understanding**: Understands natural language commands without requiring specific keywords
- **Location Guidance**: Clear information about where content is located in the document
- **Descriptive Images**: Detailed descriptions of images for BLV users
- **Keyboard Navigation**: Full keyboard support for navigation
- **Screen Reader Compatibility**: Designed to work well with screen readers

## License

MIT
