# Streamlit Deployment Guide

This guide explains how to deploy the BLV Document Chat application to Streamlit Cloud.

## Deployment Files

The following files have been created specifically for Streamlit deployment:

1. `apt.txt` - Lists the system dependencies required for audio processing
2. `requirements.txt` - Modified requirements file that removes unused dependencies (original requirements backed up as `requirements_original.txt`)
3. `app_deployment.py` - Deployment version of the main application file
4. `streamlit_app.py` - Simple entry point for Streamlit Cloud
5. `.streamlit/config.toml` - Streamlit configuration file
6. `test_deployment.sh` - Script to test deployment setup locally

## Deployment Steps

1. **Update your GitHub repository**

   Make sure all the deployment files are pushed to your GitHub repository.

2. **Create a new Streamlit app**

   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Set the main file path to `blv_doc_chat/streamlit_app.py`
   - Set the Python version to 3.9 or higher

3. **Configure environment variables**

   Add the following environment variables in the Streamlit Cloud settings:

   - `OPENAI_API_KEY` - Your OpenAI API key

4. **Advanced settings**

   - Make sure the "Enable apt-get" option is checked to allow installation of system dependencies from apt.txt

## Troubleshooting

If you encounter issues with audio dependencies:

1. Check the Streamlit logs for specific error messages
2. Verify that the apt.txt file is being properly recognized
3. Try adding additional system dependencies to apt.txt if needed

## Audio Functionality

The deployment is configured to maintain audio functionality using:

- System dependencies from apt.txt (libasound2-dev, portaudio19-dev, python3-pyaudio)
- Python audio libraries (pyaudio, sounddevice, soundfile)

## Notes

- The Hume voice interface is not used in this deployment
- All OpenAI voice functionality should work as expected
- Document processing and RAG functionality remain unchanged
