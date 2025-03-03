"""
Utility functions for OpenAI voice interface.
"""

import os
import asyncio
import json
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional, Tuple, List, Callable, Dict, Any

from utils import logger, audio_logger, get_temp_file_path

def record_with_vad(
    sample_rate: int,
    channels: int,
    threshold: float,
    silence_duration: float,
    min_speech_duration: float,
    should_stop: bool = False
) -> Optional[str]:
    """
    Record audio with voice activity detection.
    
    Args:
        sample_rate: Audio sample rate
        channels: Number of audio channels
        threshold: Energy threshold for voice activity detection
        silence_duration: Duration of silence to consider end of speech
        min_speech_duration: Minimum duration to consider as speech
        should_stop: Flag to check if recording should be stopped
        
    Returns:
        Path to recorded audio file, or None if no speech detected
    """
    try:
        # Initialize variables
        audio_buffer = []
        is_speech = False
        speech_start_time = None
        last_speech_time = None
        start_time = time.time()
        
        # Start recording
        def callback(indata, frames, time_info, status):
            if status:
                pass  # Removed audio status logging
            
            # Add data to buffer
            audio_buffer.append(indata.copy())
            
            # Calculate energy
            energy = np.mean(np.abs(indata))
            
            nonlocal is_speech, speech_start_time, last_speech_time
            
            # Voice activity detection
            if energy > threshold:
                if not is_speech:
                    is_speech = True
                    speech_start_time = time.time()
                last_speech_time = time.time()
        
        # Start recording stream
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            callback=callback
        )
        stream.start()
        
        # Wait for speech to end or timeout
        max_duration = 15  # Maximum recording duration in seconds
        try:
            while True:
                time.sleep(0.1)
                
                # Check if we should stop
                if should_stop:
                    break
                
                # Check for timeout
                current_time = time.time()
                if current_time - start_time > max_duration:
                    break
                
                # Check if speech has started
                if is_speech:
                    # Check if silence duration exceeded
                    if current_time - last_speech_time > silence_duration:
                        # Check if speech duration is long enough
                        if last_speech_time - speech_start_time >= min_speech_duration:
                            break
                        else:
                            # Reset if speech was too short
                            is_speech = False
                            audio_buffer = audio_buffer[-int(sample_rate * 0.5):]  # Keep last 0.5 seconds
        finally:
            # Stop recording
            stream.stop()
            stream.close()
        
        # Check if we have enough speech
        if is_speech and last_speech_time - speech_start_time >= min_speech_duration:
            # Combine audio buffer
            if not audio_buffer:
                return None
            
            try:
                audio_data = np.concatenate(audio_buffer)
                
                # Save audio to file
                output_path = get_temp_file_path("vad_recording", ".wav")
                sf.write(output_path, audio_data, sample_rate)
                
                return output_path
            except Exception as e:
                logger.error(f"Error processing VAD audio data: {e}")
                return None
        else:
            return None
    
    except Exception as e:
        logger.error(f"Error in VAD recording: {e}")
        return None

async def continuous_listening_worker(
    interface,
    loop: asyncio.AbstractEventLoop
) -> None:
    """
    Worker thread for continuous listening.
    
    Args:
        interface: OpenAIVoiceInterface instance
        loop: Asyncio event loop
    """
    # Set up voice activity detection parameters
    vad_threshold = 0.02  # Adjust based on testing
    silence_duration = 1.0  # Seconds of silence to consider end of speech
    min_speech_duration = 0.5  # Minimum duration to consider as speech
    
    try:
        while interface.continuous_listening and not interface.should_stop:
            try:
                # Record audio with voice activity detection
                audio_path = record_with_vad(
                    interface.sample_rate,
                    interface.channels,
                    vad_threshold,
                    silence_duration,
                    min_speech_duration,
                    interface.should_stop
                )
                
                if not audio_path or not os.path.exists(audio_path):
                    continue
                
                # Process the audio
                transcript = await interface.speech_to_text(audio_path)
                
                if not transcript:
                    continue
                
                # Process user intent
                is_command, command_type, command_content = interface._process_user_intent(transcript)
                
                # Handle commands or general transcription
                if is_command and command_type:
                    if command_type == "stop":
                        interface.stop_speaking()
                    elif command_type == "continue":
                        interface.continue_speaking()
                    elif command_type == "search" and interface.command_callback:
                        interface.command_callback("search", command_content or "")
                    elif command_type == "read_document":
                        interface.read_current_document()
                    elif command_type == "read_page":
                        interface.read_current_page()
                    elif command_type == "describe_interface":
                        interface.describe_interface()
                    elif command_type == "help":
                        interface.provide_help()
                elif interface.transcription_callback:
                    # If not a command, treat as general transcription
                    interface.transcription_callback(transcript)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in continuous listening worker: {e}")
                
                # Track errors to prevent error loops
                current_time = time.time()
                if current_time - interface.last_error_time < 5:  # Within 5 seconds
                    interface.error_count += 1
                else:
                    interface.error_count = 1
                interface.last_error_time = current_time
                
                # If too many errors in a short time, take a longer break
                if interface.error_count > 3:
                    time.sleep(5)
                else:
                    time.sleep(1)  # Prevent tight loop on error
    
    finally:
        interface.continuous_listening = False

async def continuous_listening_worker_with_audio_model(
    interface,
    loop: asyncio.AbstractEventLoop
) -> None:
    """
    Worker thread for continuous listening using the GPT-4o audio preview model.
    
    Args:
        interface: OpenAIVoiceInterface instance
        loop: Asyncio event loop
    """
    # Set up voice activity detection parameters
    vad_threshold = 0.02  # Adjust based on testing
    silence_duration = 1.0  # Seconds of silence to consider end of speech
    min_speech_duration = 0.5  # Minimum duration to consider as speech
    
    try:
        while interface.continuous_listening and not interface.should_stop:
            try:
                # Record audio with voice activity detection
                audio_path = record_with_vad(
                    interface.sample_rate,
                    interface.channels,
                    vad_threshold,
                    silence_duration,
                    min_speech_duration,
                    interface.should_stop
                )
                
                if not audio_path or not os.path.exists(audio_path):
                    continue
                
                # Process the audio with GPT-4o audio preview model
                response_path = await interface.audio_conversation(audio_path)
                
                if not response_path or not os.path.exists(response_path):
                    continue
                    
                # Play the response
                interface.play_audio(response_path, block=True)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in continuous listening worker with audio model: {e}")
                
                # Track errors to prevent error loops
                current_time = time.time()
                if current_time - interface.last_error_time < 5:  # Within 5 seconds
                    interface.error_count += 1
                else:
                    interface.error_count = 1
                interface.last_error_time = current_time
                
                # If too many errors in a short time, take a longer break
                if interface.error_count > 3:
                    time.sleep(5)
                else:
                    time.sleep(1)  # Prevent tight loop on error
    
    finally:
        interface.continuous_listening = False
