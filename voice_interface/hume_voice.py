"""
Hume voice interface for speech-to-text and text-to-speech with continuous voice guidance.
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
import websockets
from hume import HumeVoiceClient, HumeStreamClient
from hume.models.config import ProsodyConfig

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    HUME_API_KEY,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT
)
from utils import logger, get_temp_file_path, encode_audio_to_base64, decode_base64_to_audio

class HumeVoiceInterface:
    """
    Voice interface using Hume API for speech-to-text and text-to-speech.
    """
    
    def __init__(self):
        """Initialize the Hume voice interface."""
        self.client = HumeVoiceClient(api_key=HUME_API_KEY)
        self.stream_client = HumeStreamClient(api_key=HUME_API_KEY)
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
        self.voice_speed = 1.0
        self.voice_volume = 0.8
        self.voice_commands = {
            "stop": self.stop_speaking,
            "pause": self.pause_speaking,
            "continue": self.continue_speaking,
            "search for": self.handle_search_command,
            "find": self.handle_search_command,
            "help": self.provide_help,
            "read document": self.read_current_document,
            "read page": self.read_current_page,
            "describe interface": self.describe_interface,
        }
        self.command_callback = None
        self.transcription_callback = None
        logger.info("HumeVoiceInterface initialized")
    
    def set_voice_properties(self, speed: float = None, volume: float = None) -> None:
        """
        Set voice properties.
        
        Args:
            speed: Voice speed (0.5 to 2.0)
            volume: Voice volume (0.1 to 1.0)
        """
        if speed is not None:
            self.voice_speed = max(0.5, min(2.0, speed))
        
        if volume is not None:
            self.voice_volume = max(0.1, min(1.0, volume))
        
        logger.info(f"Voice properties set: speed={self.voice_speed}, volume={self.voice_volume}")
    
    def text_to_speech(self, text: str, voice_description: Optional[str] = None) -> str:
        """
        Convert text to speech using Hume TTS.
        
        Args:
            text: Text to convert to speech
            voice_description: Optional description of the voice to use
            
        Returns:
            Path to the generated audio file
        """
        logger.info(f"Converting text to speech: {text[:50]}...")
        
        try:
            # Create a temporary file to store the audio
            output_path = get_temp_file_path("tts_output", ".mp3")
            
            # Configure the voice
            voice_config = {
                "name": "serene",  # Default voice
                "settings": {
                    "stability": 0.7,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "speaker_boost": True,
                    "speed": self.voice_speed,  # Apply voice speed
                }
            }
            
            if voice_description:
                # If a voice description is provided, use it to configure the voice
                voice_config["description"] = voice_description
            
            # Use Hume's TTS API to generate speech
            response = self.client.text_to_speech(
                text=text,
                voice_config=voice_config,
                output_format="mp3"
            )
            
            # Save the audio to a file
            with open(output_path, "wb") as f:
                f.write(response.audio)
            
            logger.info(f"Text-to-speech output saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            # Create a fallback audio file with a message about the error
            try:
                output_path = get_temp_file_path("tts_error", ".wav")
                sample_rate = 44100
                duration = 2
                frequency = 440
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
                sf.write(output_path, audio_data, sample_rate)
                logger.info(f"Created error audio file: {output_path}")
                return output_path
            except:
                logger.error("Failed to create error audio file")
                raise
    
    def play_audio(self, audio_path: str, block: bool = True, volume: float = None) -> None:
        """
        Play audio file.
        
        Args:
            audio_path: Path to audio file
            block: Whether to block until audio playback is complete
        """
        logger.info(f"Playing audio: {audio_path}")
        
        try:
            # Load audio file
            data, samplerate = sf.read(audio_path)
            
            # Set speaking flag
            self.is_speaking = True
            
            # Apply volume
            volume_to_use = volume if volume is not None else self.voice_volume
            data = data * volume_to_use  # Scale amplitude by volume
            
            # Play audio
            sd.play(data, samplerate)
            
            if block:
                sd.wait()
                self.is_speaking = False
                logger.info("Audio playback complete")
            else:
                # Start a thread to monitor playback completion
                def monitor_playback():
                    sd.wait()
                    self.is_speaking = False
                    logger.info("Audio playback complete")
                
                threading.Thread(target=monitor_playback, daemon=True).start()
        except Exception as e:
            self.is_speaking = False
            logger.error(f"Error playing audio: {e}")
            raise
    
    def stop_speaking(self) -> None:
        """Stop current audio playback."""
        logger.info("Stopping audio playback")
        sd.stop()
        self.is_speaking = False
    
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
        - "Stop" or "Pause" to stop the current speech
        - "Continue" to continue with the previous information
        - "Search for..." followed by your query to search the document
        - "Find..." followed by your query to search the document
        - "Read document" to have the current document read aloud
        - "Read page" to have the current page read aloud
        - "Describe interface" to get a description of the current interface
        - "Help" to hear this help information
        
        You can also ask questions about the document at any time.
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
        if interrupt and self.is_speaking:
            self.stop_speaking()
        
        # Add to speech queue
        self.speech_queue.put(text)
        
        # If speaking thread is not running, start it
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
                
                # Convert to speech and play
                audio_path = self.text_to_speech(text)
                self.play_audio(audio_path, block=True)
                
                # Mark task as done
                self.speech_queue.task_done()
            except Exception as e:
                logger.error(f"Error in speaking worker: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def start_recording(self) -> None:
        """Start recording audio."""
        logger.info("Starting audio recording")
        
        if self.recording:
            logger.warning("Recording already in progress")
            return
        
        try:
            self.recording = True
            self.audio_buffer = []
            
            def callback(indata, frames, time, status):
                """Callback for audio recording."""
                if status:
                    logger.warning(f"Audio recording status: {status}")
                self.audio_buffer.append(indata.copy())
            
            # Start recording
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback
            )
            self.stream.start()
            
            logger.info("Audio recording started")
        except Exception as e:
            self.recording = False
            logger.error(f"Error starting audio recording: {e}")
            # Create a dummy audio buffer to prevent crashes
            self.audio_buffer = []
    
    def stop_recording(self) -> str:
        """
        Stop recording audio.
        
        Returns:
            Path to the recorded audio file
        """
        logger.info("Stopping audio recording")
        
        if not self.recording:
            logger.warning("No recording in progress")
            return ""
        
        try:
            # Stop recording
            if hasattr(self, 'stream'):
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logger.error(f"Error stopping audio stream: {e}")
            
            self.recording = False
            
            # Combine audio buffer
            if not self.audio_buffer:
                logger.warning("No audio recorded")
                return ""
            
            try:
                audio_data = np.concatenate(self.audio_buffer)
                
                # Save audio to file
                output_path = get_temp_file_path("recording", ".wav")
                sf.write(output_path, audio_data, self.sample_rate)
                
                logger.info(f"Audio recording saved to {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                # Create a dummy audio file to prevent crashes
                output_path = get_temp_file_path("dummy_recording", ".wav")
                
                # Create a simple sine wave as a placeholder
                sample_rate = 44100
                duration = 1  # seconds
                frequency = 440  # Hz (A4)
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                dummy_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
                
                # Save as WAV file
                sf.write(output_path, dummy_audio, sample_rate)
                
                logger.info(f"Created dummy audio file: {output_path}")
                return output_path
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.recording = False
            return ""
    
    async def speech_to_text(self, audio_path: str) -> str:
        """
        Convert speech to text using Hume API.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        logger.info(f"Converting speech to text: {audio_path}")
        
        try:
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            # Use Hume's speech-to-text API
            async with self.stream_client.connect() as connection:
                # Send audio data
                await connection.send_audio(audio_data)
                
                # Get transcription
                response = await connection.receive()
                
                # Extract transcript
                if hasattr(response, 'transcript') and response.transcript:
                    transcript = response.transcript
                    logger.info(f"Transcription: {transcript}")
                    return transcript
                else:
                    logger.warning("No transcript received from Hume API")
                    return "I couldn't understand what you said. Please try again."
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return f"Error in speech-to-text: {str(e)}"
    
    def start_continuous_listening(self, 
                                  transcription_callback: Callable[[str], None],
                                  command_callback: Optional[Callable[[str, str], None]] = None) -> None:
        """
        Start continuous listening for voice input.
        
        Args:
            transcription_callback: Callback function to handle transcriptions
            command_callback: Callback function to handle commands (command_type, command_text)
        """
        logger.info("Starting continuous listening")
        
        if self.continuous_listening:
            logger.warning("Continuous listening already in progress")
            return
        
        self.continuous_listening = True
        self.should_stop = False
        self.transcription_callback = transcription_callback
        self.command_callback = command_callback
        
        # Start listening thread
        self.listening_thread = threading.Thread(target=self._continuous_listening_worker, daemon=True)
        self.listening_thread.start()
        
        # Announce that voice mode is active
        self.speak("Voice mode activated. You can speak to interact with the application.")
    
    def stop_continuous_listening(self) -> None:
        """Stop continuous listening."""
        logger.info("Stopping continuous listening")
        
        if not self.continuous_listening:
            logger.warning("Continuous listening not in progress")
            return
        
        self.continuous_listening = False
        self.should_stop = True
        
        # Wait for thread to finish
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=2)
        
        logger.info("Continuous listening stopped")
    
    def _continuous_listening_worker(self) -> None:
        """Worker thread for continuous listening."""
        logger.info("Continuous listening worker started")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Set up voice activity detection parameters
        vad_threshold = 0.02  # Adjust based on testing
        silence_duration = 1.0  # Seconds of silence to consider end of speech
        min_speech_duration = 0.5  # Minimum duration to consider as speech
        
        try:
            while self.continuous_listening and not self.should_stop:
                try:
                    # Record audio with voice activity detection
                    audio_path = self._record_with_vad(vad_threshold, silence_duration, min_speech_duration)
                    
                    if not audio_path or not os.path.exists(audio_path):
                        continue
                    
                    # Process the audio
                    transcript = loop.run_until_complete(self.speech_to_text(audio_path))
                    
                    if not transcript:
                        continue
                    
                    # Check for commands
                    command_detected = False
                    for command_key, command_handler in self.voice_commands.items():
                        if transcript.lower().startswith(command_key.lower()):
                            # Extract command content
                            command_content = transcript[len(command_key):].strip()
                            
                            # Handle command
                            if command_key.lower() in ["search for", "find"]:
                                if self.command_callback:
                                    self.command_callback("search", command_content)
                            else:
                                command_handler()
                            
                            command_detected = True
                            break
                    
                    # If not a command, treat as transcription
                    if not command_detected and self.transcription_callback:
                        self.transcription_callback(transcript)
                    
                except Exception as e:
                    logger.error(f"Error in continuous listening worker: {e}")
                    time.sleep(1)  # Prevent tight loop on error
        
        finally:
            logger.info("Continuous listening worker stopped")
            self.continuous_listening = False
    
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
                    logger.warning(f"Audio recording status: {status}")
                
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
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback
            )
            stream.start()
            
            # Wait for speech to end or timeout
            max_duration = 30  # Maximum recording duration in seconds
            try:
                while True:
                    time.sleep(0.1)
                    
                    # Check if we should stop
                    if self.should_stop:
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
                                audio_buffer = audio_buffer[-int(self.sample_rate * 0.5):]  # Keep last 0.5 seconds
            finally:
                # Stop recording
                stream.stop()
                stream.close()
            
            # Check if we have enough speech
            if is_speech and last_speech_time - speech_start_time >= min_speech_duration:
                # Combine audio buffer
                if not audio_buffer:
                    logger.warning("No audio recorded")
                    return None
                
                try:
                    audio_data = np.concatenate(audio_buffer)
                    
                    # Save audio to file
                    output_path = get_temp_file_path("vad_recording", ".wav")
                    sf.write(output_path, audio_data, self.sample_rate)
                    
                    logger.info(f"VAD recording saved to {output_path}")
                    return output_path
                except Exception as e:
                    logger.error(f"Error processing VAD audio data: {e}")
                    return None
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error in VAD recording: {e}")
            return None
    
    async def start_voice_chat(self, on_transcription: Callable[[str], None]) -> None:
        """
        Start a voice chat session.
        
        Args:
            on_transcription: Callback function to handle transcriptions
        """
        logger.info("Starting voice chat session")
        
        try:
            # Start continuous listening
            self.start_continuous_listening(on_transcription)
            
            # Wait for user to stop the session
            while self.continuous_listening and not self.should_stop:
                await asyncio.sleep(1)
            
            logger.info("Voice chat session ended")
        except Exception as e:
            logger.error(f"Error in voice chat: {e}")
            self.stop_continuous_listening()
    
    async def stream_text_to_speech(self, text: str, chunk_size: int = 100) -> AsyncGenerator[str, None]:
        """
        Stream text to speech in chunks.
        
        Args:
            text: Text to convert to speech
            chunk_size: Number of characters per chunk
            
        Yields:
            Paths to audio files for each chunk
        """
        logger.info(f"Streaming text to speech: {text[:50]}...")
        
        try:
            # Split text into chunks
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Convert each chunk to speech
            for i, chunk in enumerate(chunks):
                logger.info(f"Converting chunk {i+1}/{len(chunks)}")
                
                # Convert chunk to speech
                audio_path = self.text_to_speech(chunk)
                
                # Yield audio path
                yield audio_path
                
                # Simulate processing time
                await asyncio.sleep(0.1)
            
            logger.info("Text-to-speech streaming complete")
        except Exception as e:
            logger.error(f"Error in text-to-speech streaming: {e}")
            raise
    
    def describe_ui_element(self, element_type: str, element_content: str) -> str:
        """
        Generate a description of a UI element for screen reading.
        
        Args:
            element_type: Type of UI element (e.g., button, input, header)
            element_content: Content of the UI element
            
        Returns:
            Description of the UI element
        """
        descriptions = {
            "button": f"Button: {element_content}",
            "input": f"Text input field: {element_content}",
            "header": f"Heading: {element_content}",
            "checkbox": f"Checkbox: {element_content}",
            "selectbox": f"Dropdown menu: {element_content}",
            "slider": f"Slider: {element_content}",
            "radio": f"Radio button: {element_content}",
            "text": element_content,
            "link": f"Link: {element_content}",
            "image": f"Image: {element_content}",
            "file_uploader": f"File upload area: {element_content}",
            "error": f"Error message: {element_content}",
            "success": f"Success message: {element_content}",
            "info": f"Information: {element_content}",
            "warning": f"Warning: {element_content}",
        }
        
        return descriptions.get(element_type.lower(), f"{element_type}: {element_content}")
    
    def announce_page_content(self, elements: List[Tuple[str, str]]) -> None:
        """
        Announce the content of a page.
        
        Args:
            elements: List of (element_type, element_content) tuples
        """
        descriptions = []
        
        for element_type, element_content in elements:
            descriptions.append(self.describe_ui_element(element_type, element_content))
        
        # Join descriptions and speak
        self.speak("\n".join(descriptions))
    
    def announce_navigation(self, location: str) -> None:
        """
        Announce navigation to a new location.
        
        Args:
            location: Name of the location
        """
        self.speak(f"Navigated to {location}")
    
    def announce_action(self, action: str, result: str) -> None:
        """
        Announce the result of an action.
        
        Args:
            action: Action performed
            result: Result of the action
        """
        self.speak(f"{action}: {result}")
