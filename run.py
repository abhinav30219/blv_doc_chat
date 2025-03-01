#!/usr/bin/env python3
"""
Run script for the BLV Document Chat application.
"""

import os
import sys
import subprocess

def main():
    """Run the BLV Document Chat application."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Error: .env file not found.")
        print("Please create a .env file from the template:")
        print("cp .env.template .env")
        print("Then edit the .env file to add your API keys.")
        return 1
    
    # Run the application using streamlit
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running streamlit: {e}")
        return 1
    except FileNotFoundError:
        print("Error: streamlit not found.")
        print("Please install streamlit:")
        print("pip install streamlit")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
