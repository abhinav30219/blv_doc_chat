# Streamlit Deployment Guide for BLV Document Chat

This guide explains how to deploy the simplified version of the BLV Document Chat application to Streamlit Cloud.

## Changes Made for Deployment

The following changes have been made to ensure successful deployment on Streamlit:

1. **Removed LightRAG Dependencies**: The application now uses SimpleRAG instead of LightRAG to avoid dependency issues.
2. **Added `pipmaster` Dependency**: Added to requirements.txt to support the application.
3. **Fixed OpenAI Assistants API**: Updated the API calls to use the new format with `tool_resources`.

## Deployment Steps

### 1. Prepare Your Repository

Make sure your GitHub repository includes all the necessary files:
- All Python code files
- `requirements.txt` with updated dependencies
- `packages.txt` for system dependencies
- `.streamlit/config.toml` for Streamlit configuration
- `streamlit_app.py` as the entry point

### 2. Create a New Streamlit App

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Set the main file path to `blv_doc_chat/streamlit_app.py`
4. Set the Python version to 3.9 or higher

### 3. Configure Environment Variables

Add the following environment variables in the Streamlit Cloud settings (Advanced Settings → Secrets):

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_TTS_MODEL = "tts-1-hd"
OPENAI_STT_MODEL = "whisper-1"
OPENAI_ASSISTANT_MODEL = "gpt-4o"
VOICE_STYLE = "sage"
VOICE_ENABLED_BY_DEFAULT = "true"
OPENAI_FILE_ENABLED = "true"
```

### 4. Enable apt-get

In the Advanced Settings, make sure to check the "Enable apt-get" option. This will allow Streamlit to install the system dependencies listed in your `packages.txt` file:
- libasound2-dev
- portaudio19-dev
- python3-pyaudio

These packages are necessary for the audio processing functionality in your application.

## Troubleshooting

If you encounter issues with the deployment:

1. **Check Streamlit Logs**: Look at the logs in the Streamlit Cloud dashboard for specific error messages.

2. **Dependency Issues**: If you see errors related to missing dependencies, you may need to add them to `requirements.txt`.

3. **System Dependency Issues**: If you see errors related to system libraries, make sure your `packages.txt` file is correctly formatted and that "Enable apt-get" is checked.

4. **OpenAI API Issues**: Verify that your OpenAI API key is correctly set in the Streamlit secrets.

5. **Memory or Resource Issues**: If the app crashes due to memory limits, you might need to optimize your code or upgrade your Streamlit plan.

## Limitations

This simplified deployment has the following limitations:

1. It uses SimpleRAG instead of LightRAG, which may affect performance for large document collections.
2. Audio processing might be limited by the Streamlit environment.

## Testing Locally

To test the deployment configuration locally before deploying to Streamlit Cloud:

```bash
cd blv_doc_chat
streamlit run streamlit_app.py
```

This will help you identify any issues before deploying to Streamlit Cloud.
