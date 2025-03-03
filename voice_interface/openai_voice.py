"""
OpenAI voice interface for speech-to-text and text-to-speech with continuous voice guidance.
"""

import os
import asyncio
import json
import base64
import tempfile
import threading
import time
import queue
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator, Tuple
import sounddevice as sd
import soundfile as sf
import numpy as np
import openai
from openai import OpenAI

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_API_KEY,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
    VOICE_SPEED,
    VOICE_VOLUME,
    OPENAI_AUDIO_MODEL,
    OPENAI_TTS_MODEL,
    OPENAI_STT_MODEL,
    OPENAI_CHAT_MODEL,
    USE_AUDIO_MODEL
)
from utils import logger, audio_logger, get_temp_file_path, encode_audio_to_base64, decode_base64_to_audio
from voice_interface.openai_voice_utils import (
    record_with_vad,
    continuous_listening_worker,
    continuous_listening_worker_with_audio_model
)

class OpenAIVoiceInterface:
    """
    Voice interface using OpenAI API for speech-to-text and text-to-speech.
    """
    
    def __init__(self):
        """Initialize the OpenAI voice interface."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.channels = AUDIO_CHANNELS
        self.recording = False
        self.continuous_listening = False
        self.audio_buffer = []
        self.speech_queue = queue.Queue()
        self.listening_thread = None
        self.speaking_thread = None
        self.is_speaking = False
        self.should_stop = False
        self.voice_speed = VOICE_SPEED
        self.voice_volume = VOICE_VOLUME
        self.voice_style = "sage"  # Default voice style
        self.conversation_history = []  # Store conversation history for context
        self.command_callback = None
        self.transcription_callback = None
        
        # Add locks for thread safety
        self.speaking_lock = threading.Lock()
        self.recording_lock = threading.Lock()
        self.audio_processing_lock = threading.Lock()
        
        # Add state tracking
        self.active_audio_paths = set()
        self.last_error_time = 0
        self.error_count = 0
        self.last_speech_time = 0
        
        logger.info("OpenAIVoiceInterface initialized")
    
    def set_voice_properties(self, speed: float = None, volume: float = None, style: str = None) -> None:
        """
        Set voice properties.
        
        Args:
            speed: Voice speed (0.5 to 2.0)
            volume: Voice volume (0.1 to 1.0)
            style: Voice style (e.g., "sage", "alloy", "echo", etc.)
        """
        if speed is not None:
            self.voice_speed = max(0.5, min(2.0, speed))
        
        if volume is not None:
            self.voice_volume = max(0.1, min(1.0, volume))
        
        if style is not None:
            self.voice_style = style
    
    def text_to_speech(self, text: str, voice_style: Optional[str] = None) -> str:
        """
        Convert text to speech using OpenAI TTS.
        
        Args:
            text: Text to convert to speech
            voice_style: Optional voice style to use
            
        Returns:
            Path to the generated audio file
        """
        # Skip empty text
        if not text or not text.strip():
            return ""
            
        # Sanitize text to prevent issues
        text = text.strip()
        
        # Limit text length to prevent very long audio generation
        if len(text) > 5000:
            text = text[:4997] + "..."
            
        try:
            # Create a temporary file to store the audio
            output_path = get_temp_file_path("tts_output", f".{AUDIO_FORMAT}")
            
            # Use the specified voice style or the default
            voice = voice_style if voice_style else self.voice_style
            
            # Use OpenAI's TTS API to generate speech
            response = self.client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=text,
                speed=self.voice_speed,
                response_format=AUDIO_FORMAT
            )
            
            # Save the audio to a file
            response.stream_to_file(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            
            # Instead of creating a beeping sound, return a spoken error message
            try:
                # Only create error audio if we haven't had too many errors recently
                current_time = time.time()
                if current_time - self.last_error_time > 10:  # Only create error audio every 10 seconds max
                    self.last_error_time = current_time
                    self.error_count = 1
                    
                    # Create a simple error message
                    error_message = "I'm sorry, there was an error generating speech."
                    
                    # Try to use a simpler TTS call with minimal parameters
                    try:
                        output_path = get_temp_file_path("tts_error_message", f".{AUDIO_FORMAT}")
                        simple_response = self.client.audio.speech.create(
                            model="tts-1",  # Use simpler model
                            voice="alloy",  # Use default voice
                            input=error_message,
                            response_format=AUDIO_FORMAT
                        )
                        simple_response.stream_to_file(output_path)
                        return output_path
                    except Exception as inner_e:
                        # If that fails too, return empty string instead of beeping
                        return ""
                else:
                    # Too many errors recently, just return empty to avoid error spam
                    self.error_count += 1
                    return ""
            except Exception as fallback_e:
                return ""
    
    def play_audio(self, audio_path: str, block: bool = True, volume: float = None) -> None:
        """
        Play audio file.
        
        Args:
            audio_path: Path to audio file
            block: Whether to block until audio playback is complete
            volume: Optional volume override
        """
        # Skip if no audio path or file doesn't exist
        if not audio_path or not os.path.exists(audio_path):
            return
            
        # Use lock to prevent multiple simultaneous playbacks
        with self.speaking_lock:
            try:
                # Stop any current playback before starting new one
                if self.is_speaking:
                    self.stop_speaking()
                    # Small delay to ensure previous audio is fully stopped
                    time.sleep(0.1)
                
                # Load audio file
                data, samplerate = sf.read(audio_path)
                
                # Set speaking flag
                self.is_speaking = True
                self.last_speech_time = time.time()
                
                # Track this audio path
                self.active_audio_paths.add(audio_path)
                
                # Apply volume
                volume_to_use = volume if volume is not None else self.voice_volume
                data = data * volume_to_use  # Scale amplitude by volume
                
                # Play audio
                sd.play(data, samplerate)
                
                if block:
                    sd.wait()
                    with self.speaking_lock:
                        self.is_speaking = False
                        self.active_audio_paths.discard(audio_path)
                else:
                    # Start a thread to monitor playback completion
                    def monitor_playback():
                        try:
                            sd.wait()
                        except Exception as e:
                            logger.error(f"Error waiting for audio playback: {e}")
                        finally:
                            with self.speaking_lock:
                                self.is_speaking = False
                                self.active_audio_paths.discard(audio_path)
                    
                    thread = threading.Thread(target=monitor_playback, daemon=True)
                    thread.start()
            except Exception as e:
                with self.speaking_lock:
                    self.is_speaking = False
                    self.active_audio_paths.discard(audio_path)
                logger.error(f"Error playing audio: {e}")
                # Don't raise the exception, just log it to prevent crashes
    
    def stop_speaking(self) -> None:
        """Stop current audio playback."""
        with self.speaking_lock:
            sd.stop()
            self.is_speaking = False
            self.active_audio_paths.clear()
    
    def pause_speaking(self) -> None:
        """Pause current audio playback (alias for stop_speaking)."""
        self.stop_speaking()
    
    def continue_speaking(self) -> None:
        """Continue speaking (placeholder for now)."""
        self.speak("Continuing with the previous information.")
    
    def handle_search_command(self, query: str) -> None:
        """Handle search command."""
        if self.command_callback:
            self.command_callback("search", query)
    
    def provide_help(self) -> None:
        """Provide help information."""
        help_text = """
        You can use the following voice commands:
        - Ask me to stop or pause when you want me to stop talking
        - Ask me to continue if you want me to continue
        - Ask me to search for something in your document
        - Ask me to read the document or a specific page
        - Ask me to describe the interface
        - Ask for help anytime you need assistance
        
        You can also ask questions about your document at any time.
        """
        self.speak(help_text)
    
    def read_current_document(self) -> None:
        """Read the current document."""
        if self.command_callback:
            self.command_callback("read_document", "")
        else:
            self.speak("I'm sorry, I can't read the document at this time.")
    
    def read_current_page(self) -> None:
        """Read the current page."""
        if self.command_callback:
            self.command_callback("read_page", "")
        else:
            self.speak("I'm sorry, I can't read the page at this time.")
    
    def describe_interface(self) -> None:
        """Describe the current interface."""
        if self.command_callback:
            self.command_callback("describe_interface", "")
        else:
            self.speak("I'm sorry, I can't describe the interface at this time.")
    
    def speak(self, text: str, interrupt: bool = True) -> None:
        """
        Speak text using text-to-speech.
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
        """
        # Skip empty text
        if not text or not text.strip():
            return
            
        # Sanitize text to prevent issues
        text = text.strip()
        
        # Limit text length to prevent very long audio generation
        if len(text) > 5000:
            text = text[:4997] + "..."
        
        # Check if we should interrupt current speech
        if interrupt and self.is_speaking:
            self.stop_speaking()
            # Small delay to ensure previous audio is fully stopped
            time.sleep(0.1)
        
        # Rate limit speech to prevent too many queued items
        current_time = time.time()
        if current_time - self.last_speech_time < 0.5 and not self.speech_queue.empty():
            # If we've spoken very recently and queue is not empty, 
            # combine with the last queued item instead of adding a new one
            try:
                # Get the last item without removing it
                last_items = []
                while not self.speech_queue.empty():
                    last_items.append(self.speech_queue.get())
                
                # Combine the last item with the new text
                if last_items:
                    combined_text = last_items[-1] + " " + text
                    last_items[-1] = combined_text
                else:
                    last_items.append(text)
                
                # Put all items back
                for item in last_items:
                    self.speech_queue.put(item)
            except Exception as e:
                # If combining fails, just add as new item
                logger.error(f"Error combining speech text: {e}")
                self.speech_queue.put(text)
        else:
            # Add to speech queue as normal
            self.speech_queue.put(text)
        
        self.last_speech_time = current_time
        
        # If speaking thread is not running, start it
        with self.speaking_lock:
            if not self.speaking_thread or not self.speaking_thread.is_alive():
                self.speaking_thread = threading.Thread(target=self._speaking_worker, daemon=True)
                self.speaking_thread.start()
    
    def _speaking_worker(self) -> None:
        """Worker thread for speaking text from the queue."""
        while not self.should_stop:
            try:
                # Get text from queue with timeout
                try:
                    text = self.speech_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Skip empty text
                if not text or not text.strip():
                    self.speech_queue.task_done()
                    continue
                
                # Check if we should stop
                if self.should_stop:
                    break
                
                # Convert to speech and play
                try:
                    audio_path = self.text_to_speech(text)
                    if audio_path and os.path.exists(audio_path):
                        self.play_audio(audio_path, block=True)
                except Exception as e:
                    logger.error(f"Error converting text to speech: {e}")
                    # Don't try to play audio if TTS failed
                
                # Mark task as done
                self.speech_queue.task_done()
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in speaking worker: {e}")
                
                # Track errors to prevent error loops
                current_time = time.time()
                if current_time - self.last_error_time < 5:  # Within 5 seconds
                    self.error_count += 1
                else:
                    self.error_count = 1
                self.last_error_time = current_time
                
                # If too many errors in a short time, take a longer break
                if self.error_count > 3:
                    time.sleep(5)
                else:
                    time.sleep(1)  # Prevent tight loop on error
    
    def start_recording(self) -> None:
        """Start recording audio."""
        # Use lock to prevent multiple simultaneous recordings
        with self.recording_lock:
            if self.recording:
                return
            
            try:
                # Initialize variables
                self.recording = True
                self.audio_buffer = []
                
                def callback(indata, frames, time, status):
                    """Callback for audio recording."""
                    try:
                        if status:
                            pass
                        self.audio_buffer.append(indata.copy())
                    except Exception as cb_e:
                        logger.error(f"Error in recording callback: {cb_e}")
                        # Don't crash, just log the error
                
                # Test audio device before starting
                try:
                    sd.check_input_settings(
                        device=None,  # Use default device
                        channels=self.channels,
                        samplerate=self.sample_rate
                    )
                except Exception as device_e:
                    logger.error(f"Audio input device check failed: {device_e}")
                    # Try to continue anyway with default settings
                
                # Start recording with robust error handling
                try:
                    self.stream = sd.InputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        callback=callback
                    )
                    self.stream.start()
                except Exception as stream_e:
                    logger.error(f"Error starting audio stream: {stream_e}")
                    self.recording = False
                    self.audio_buffer = []
                    raise  # Re-raise to be caught by outer try-except
                
            except Exception as e:
                self.recording = False
                logger.error(f"Error starting audio recording: {e}")
                # Create a dummy audio buffer to prevent crashes
                self.audio_buffer = []
                # Return gracefully instead of crashing
    
    def stop_recording(self) -> str:
        """
        Stop recording audio.
        
        Returns:
            Path to the recorded audio file
        """
        # Use lock to prevent race conditions
        with self.recording_lock:
            if not self.recording:
                return ""
            
            # Initialize variables
            output_path = ""
            stream_closed = False
            
            try:
                # Stop recording
                if hasattr(self, 'stream'):
                    try:
                        self.stream.stop()
                        self.stream.close()
                        stream_closed = True
                    except Exception as e:
                        logger.error(f"Error stopping audio stream: {e}")
                        # Continue anyway to try to save any recorded data
                
                # Always set recording to false to prevent getting stuck
                self.recording = False
                
                # Combine audio buffer with robust error handling
                if not self.audio_buffer or len(self.audio_buffer) == 0:
                    return ""
                
                try:
                    # Check if we have valid data
                    if all(isinstance(chunk, np.ndarray) for chunk in self.audio_buffer):
                        audio_data = np.concatenate(self.audio_buffer)
                        
                        # Verify audio data is valid
                        if audio_data.size > 0:
                            # Save audio to file
                            output_path = get_temp_file_path("recording", ".wav")
                            sf.write(output_path, audio_data, self.sample_rate)
                            
                            return output_path
                        else:
                            return ""
                    else:
                        return ""
                except Exception as e:
                    logger.error(f"Error processing audio data: {e}")
                    # Return empty string instead of creating a dummy file
                    return ""
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
                self.recording = False
                return ""
            finally:
                # Clean up resources
                if hasattr(self, 'stream') and not stream_closed:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass  # Ignore errors in cleanup
    
    async def speech_to_text(self, audio_path: str) -> str:
        """
        Convert speech to text using OpenAI Whisper API.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Skip if no audio path or file doesn't exist
        if not audio_path or not os.path.exists(audio_path):
            return ""
            
        # Use lock to prevent multiple simultaneous API calls
        async with asyncio.Lock():
            try:
                # Use OpenAI's speech-to-text API
                with open(audio_path, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                
                # Extract transcript
                if transcript and hasattr(transcript, 'text'):
                    transcribed_text = transcript.text
                    return transcribed_text
                else:
                    return ""
            except Exception as e:
                logger.error(f"Error in speech-to-text: {e}")
                return ""
    
    async def audio_conversation(self, audio_path: str) -> str:
        """
        Process audio with GPT-4o audio preview model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to the response audio file
        """
        try:
            # Check if audio file exists
            if not audio_path or not os.path.exists(audio_path):
                return ""
            
            # First, get the transcript to add to conversation history
            transcript = await self.speech_to_text(audio_path)
            
            if not transcript:
                return ""
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": transcript})
            
            # Limit conversation history to last 10 messages to prevent context overflow
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Use GPT-4o to generate a response
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": """
                    You are a helpful assistant for a blind or visually impaired user who is using a document reading application.
                    Provide clear, concise responses that are informative and helpful.
                    The user may ask questions about documents they're reading, or request help with the interface.
                    """},
                    *self.conversation_history
                ]
            )
            
            # Extract response text
            if response and response.choices and response.choices[0].message:
                response_text = response.choices[0].message.content
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Convert response to speech
                response_path = self.text_to_speech(response_text)
                
                return response_path
            else:
                return ""
        except Exception as e:
            logger.error(f"Error in audio conversation: {e}")
            return ""
    
    def _process_user_intent(self, transcript: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process user intent using GPT-4o to understand natural language commands.
        
        Args:
            transcript: Transcribed text from user
            
        Returns:
            Tuple of (is_command, command_type, command_content)
        """
        # Skip empty transcript
        if not transcript or not transcript.strip():
            return False, None, None
            
        try:
            # Use GPT-4o to understand the intent
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                    You are an intent classifier for a voice interface. Analyze the user's statement and determine if it's:
                    1. A command to stop or pause speaking
                    2. A command to continue speaking
                    3. A search request
                    4. A request to read the document
                    5. A request to read a specific page
                    6. A request to describe the interface
                    7. A request for help
                    8. A general question (not a command)
                    
                    Respond with a JSON object with the following structure:
                    {
                        "is_command": true/false,
                        "command_type": "stop", "continue", "search", "read_document", "read_page", "describe_interface", "help", or null,
                        "command_content": "any specific content for the command" or null
                    }
                    """},
                    {"role": "user", "content": transcript}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            return (
                result.get("is_command", False),
                result.get("command_type"),
                result.get("command_content")
            )
        except Exception as e:
            logger.error(f"Error processing user intent: {e}")
            return False, None, None
    
    def start_continuous_listening(self, 
                                  transcription_callback: Callable[[str], None],
                                  command_callback: Optional[Callable[[str, str], None]] = None) -> None:
        """
        Start continuous listening for voice input.
        
        Args:
            transcription_callback: Callback function to handle transcriptions
            command_callback: Callback function to handle commands (command_type, command_text)
        """
        # Use lock to prevent multiple starts
        with self.recording_lock:
            if self.continuous_listening:
                return
            
            # Reset state
            self.continuous_listening = True
            self.should_stop = False
            self.transcription_callback = transcription_callback
            self.command_callback = command_callback
            
            # Start listening thread based on configuration
            if USE_AUDIO_MODEL:
                self.listening_thread = threading.Thread(
                    target=self._continuous_listening_worker_with_audio_model_wrapper, 
                    daemon=True
                )
            else:
                self.listening_thread = threading.Thread(
                    target=self._continuous_listening_worker_wrapper, 
                    daemon=True
                )
            
            self.listening_thread.start()
            
            # Announce that voice mode is active
            self.speak("Voice mode activated. You can speak to interact with the application.")
    
    def stop_continuous_listening(self) -> None:
        """Stop continuous listening."""
        # Use lock to prevent race conditions
        with self.recording_lock:
            if not self.continuous_listening:
                return
            
            self.continuous_listening = False
            self.should_stop = True
            
            # Wait for thread to finish
            if self.listening_thread and self.listening_thread.is_alive():
                self.listening_thread.join(timeout=2)
    
    def _record_with_vad(self, threshold: float, silence_duration: float, min_speech_duration: float) -> Optional[str]:
        """
        Record audio with voice activity detection.
        
        Args:
            threshold: Energy threshold for voice activity detection
            silence_duration: Duration of silence to consider end of speech
            min_speech_duration: Minimum duration to consider as speech
            
        Returns:
            Path to recorded audio file, or None if no speech detected
        """
        # Use the utility function from openai_voice_utils.py
        return record_with_vad(
            self.sample_rate,
            self.channels,
            threshold,
            silence_duration,
            min_speech_duration,
            self.should_stop
        )
    
    def _continuous_listening_worker_wrapper(self) -> None:
        """Wrapper for continuous_listening_worker to handle asyncio setup."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(continuous_listening_worker(self, loop))
        except Exception as e:
            logger.error(f"Error in continuous listening worker wrapper: {e}")
        finally:
            loop.close()
    
    def _continuous_listening_worker_with_audio_model_wrapper(self) -> None:
        """Wrapper for continuous_listening_worker_with_audio_model to handle asyncio setup."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(continuous_listening_worker_with_audio_model(self, loop))
        except Exception as e:
            logger.error(f"Error in continuous listening worker with audio model wrapper: {e}")
        finally:
            loop.close()
    
    def announce_page_content(self, elements: List[Tuple[str, str]]) -> None:
        """
        Announce page content for screen reader functionality.
        
        Args:
            elements: List of (element_type, content) tuples
        """
        # Format elements into readable text
        announcement_text = []
        
        for element_type, content in elements:
            if element_type == "header":
                # Format headers with emphasis
                announcement_text.append(f"Heading: {content}")
            elif element_type == "text":
                # Regular text
                announcement_text.append(content)
            elif element_type == "user":
                # User messages
                announcement_text.append(f"You said: {content}")
            elif element_type == "assistant":
                # Assistant messages
                announcement_text.append(f"Assistant response: {content}")
            elif element_type == "system":
                # System messages
                announcement_text.append(f"System message: {content}")
            else:
                # Other elements
                announcement_text.append(content)
        
        # Join all elements with pauses between them
        full_announcement = ". ".join(announcement_text)
        
        # Speak the announcement
        self.speak(full_announcement, interrupt=True)
