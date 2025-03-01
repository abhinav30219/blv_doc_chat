#!/usr/bin/env python3
"""
Test script to check if the imports work.
"""

import sys

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} import successful")
        return True
    except ImportError as e:
        print(f"❌ {module_name} import failed: {e}")
        return False

def main():
    """Test imports."""
    modules = [
        "openai",
        "hume",
        "lightrag",
        "streamlit",
        "langchain",
        "langchain_openai",
        "pydantic",
        "fastapi",
        "uvicorn",
        "unstructured",
        "pypdf",
        "docx",
        "pytesseract",
        "pdf2image",
        "PIL",
        "websockets",
        "sounddevice",
        "soundfile",
        "numpy",
        "pandas",
        "tqdm",
    ]
    
    success_count = 0
    for module in modules:
        if test_import(module):
            success_count += 1
    
    print(f"\nSuccessfully imported {success_count}/{len(modules)} modules")

if __name__ == "__main__":
    main()
