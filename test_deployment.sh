#!/bin/bash
# Script to test the Streamlit deployment setup locally

echo "Testing Streamlit deployment setup..."

# Check if apt.txt exists
if [ ! -f "apt.txt" ]; then
    echo "Error: apt.txt not found!"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Check if app_deployment.py exists
if [ ! -f "app_deployment.py" ]; then
    echo "Error: app_deployment.py not found!"
    exit 1
fi

# Check if streamlit_app.py exists
if [ ! -f "streamlit_app.py" ]; then
    echo "Error: streamlit_app.py not found!"
    exit 1
fi

echo "All deployment files found."

# Install system dependencies from apt.txt (if on Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies from apt.txt..."
    sudo apt-get update
    sudo apt-get install -y $(cat apt.txt)
else
    echo "Warning: apt-get not found. Skipping system dependencies installation."
    echo "You may need to install the dependencies manually:"
    cat apt.txt
fi

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv_deployment
source venv_deployment/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run streamlit_app.py

# Deactivate virtual environment when done
deactivate
