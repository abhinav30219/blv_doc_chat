"""
Utility functions for the BLV Document Chat application.
"""

import os
import logging
import json
import base64
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from config import LOG_LEVEL, LOG_FILE, TEMP_DIR, OUTPUT_DIR, PIPELINE_LOG_ENABLED, AUDIO_LOGGING_ENABLED

# Set up logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Set up pipeline-specific logger
def setup_pipeline_logger():
    """Set up pipeline-specific logger for tracking document and query processing."""
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.INFO)
    
    # Create a formatter that makes pipeline stages stand out
    formatter = logging.Formatter('%(asctime)s - [PIPELINE] %(message)s')
    
    # Add handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set up filter to only show relevant logs
    class RelevantLogFilter(logging.Filter):
        def filter(self, record):
            # Always show document processing logs
            if "document processing" in record.getMessage().lower():
                return True
            
            # Show logs related to query processing
            if "query" in record.getMessage().lower():
                return True
                
            # Show logs related to document selection
            if "document" in record.getMessage().lower() and ("select" in record.getMessage().lower() or "restrict" in record.getMessage().lower()):
                return True
                
            # Show logs related to errors
            if "error" in record.getMessage().lower() or "fail" in record.getMessage().lower():
                return True
                
            # Filter out other logs
            return False
    
    # Add filter to console handler
    console_handler.addFilter(RelevantLogFilter())
    pipeline_logger.addHandler(console_handler)
    
    # Add file handler (without filter to keep all logs in the file)
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)
    
    return pipeline_logger

# Set up audio logger (disabled by default)
def setup_audio_logger():
    """Set up audio-specific logger."""
    audio_logger = logging.getLogger("audio")
    
    if not AUDIO_LOGGING_ENABLED:
        audio_logger.setLevel(logging.CRITICAL)  # Effectively disable logging
        return audio_logger
    
    audio_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [AUDIO] %(message)s')
    
    # Add handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    audio_logger.addHandler(console_handler)
    
    return audio_logger

# Initialize loggers
logger = setup_logging()
pipeline_logger = setup_pipeline_logger() if PIPELINE_LOG_ENABLED else None
audio_logger = setup_audio_logger()

# Pipeline timing utilities
class PipelineTimer:
    """Utility class for timing pipeline stages."""
    
    def __init__(self, stage_name):
        self.stage_name = stage_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if pipeline_logger:
            pipeline_logger.info(f"Starting: {self.stage_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if pipeline_logger:
                pipeline_logger.info(f"Completed: {self.stage_name} (took {elapsed:.2f}s)")

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [TEMP_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

# File handling utilities
def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    return os.path.splitext(file_path)[1].lower()

def get_temp_file_path(prefix: str, extension: str) -> str:
    """Generate a temporary file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(TEMP_DIR, f"{prefix}_{timestamp}{extension}")

def get_output_file_path(prefix: str, extension: str) -> str:
    """Generate an output file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{prefix}_{timestamp}{extension}")

# Data handling utilities
def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data as JSON to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON data to {file_path}")

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON data from {file_path}")
    return data

# Audio utilities
def encode_audio_to_base64(audio_file_path: str) -> str:
    """Encode audio file to base64."""
    with open(audio_file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def decode_base64_to_audio(base64_string: str, output_file_path: str) -> str:
    """Decode base64 string to audio file."""
    with open(output_file_path, "wb") as audio_file:
        audio_file.write(base64.b64decode(base64_string))
    logger.info(f"Decoded base64 to audio file: {output_file_path}")
    return output_file_path

# Document structure utilities
def create_document_metadata(
    title: str,
    file_path: str,
    num_pages: int,
    sections: List[Dict[str, Any]],
    images: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create metadata for a document."""
    return {
        "title": title,
        "file_path": file_path,
        "num_pages": num_pages,
        "sections": sections,
        "images": images,
        "processed_at": datetime.now().isoformat()
    }

def format_location_reference(location: Dict[str, Any]) -> str:
    """Format a location reference for output to the user."""
    if "page" in location and "paragraph" in location:
        return f"page {location['page']}, paragraph {location['paragraph']}"
    elif "page" in location:
        return f"page {location['page']}"
    elif "section" in location and "subsection" in location:
        return f"section {location['section']}, subsection {location['subsection']}"
    elif "section" in location:
        return f"section {location['section']}"
    else:
        return "unknown location"
