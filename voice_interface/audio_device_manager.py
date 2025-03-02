"""
Audio device manager for handling audio device selection and error handling.
"""

import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Try to import alternative audio libraries
try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except ImportError:
    SOUNDCARD_AVAILABLE = False

try:
    import miniaudio
    MINIAUDIO_AVAILABLE = True
except ImportError:
    MINIAUDIO_AVAILABLE = False

from utils import logger
from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_INPUT_DEVICE,
    AUDIO_OUTPUT_DEVICE,
    AUDIO_DEVICE_RETRY_COUNT,
    AUDIO_DEVICE_RETRY_DELAY
)

class AudioDeviceManager:
    """
    Manages audio devices and provides fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the audio device manager."""
        self.input_device = AUDIO_INPUT_DEVICE
        self.output_device = AUDIO_OUTPUT_DEVICE
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.channels = AUDIO_CHANNELS
        
        # Create locks for thread safety
        self.input_device_lock = threading.Lock()
        self.output_device_lock = threading.Lock()
        
        # Initialize device lists
        self.input_devices = []
        self.output_devices = []
        self.refresh_device_lists()
        
        # Set default devices if not specified
        if self.input_device is None:
            self.input_device = sd.default.device[0]
        if self.output_device is None:
            self.output_device = sd.default.device[1]
        
        logger.info("AudioDeviceManager initialized")
        logger.info(f"Using input device: {self.input_device}")
        logger.info(f"Using output device: {self.output_device}")
    
    def refresh_device_lists(self):
        """Refresh the lists of available audio devices."""
        try:
            # Get device info from sounddevice
            devices = sd.query_devices()
            
            # Filter input and output devices
            self.input_devices = [d for d in devices if d['max_input_channels'] > 0]
            self.output_devices = [d for d in devices if d['max_output_channels'] > 0]
            
            logger.info(f"Found {len(self.input_devices)} input devices and {len(self.output_devices)} output devices")
            
            # Log device names
            logger.info("Input devices:")
            for i, device in enumerate(self.input_devices):
                logger.info(f"  {i}: {device['name']}")
            
            logger.info("Output devices:")
            for i, device in enumerate(self.output_devices):
                logger.info(f"  {i}: {device['name']}")
            
            # Also try to get devices from soundcard if available
            if SOUNDCARD_AVAILABLE:
                sc_speakers = sc.all_speakers()
                sc_microphones = sc.all_microphones()
                
                logger.info(f"Soundcard found {len(sc_speakers)} speakers and {len(sc_microphones)} microphones")
                
                # Log soundcard device names
                logger.info("Soundcard microphones:")
                for i, mic in enumerate(sc_microphones):
                    logger.info(f"  {i}: {mic.name}")
                
                logger.info("Soundcard speakers:")
                for i, speaker in enumerate(sc_speakers):
                    logger.info(f"  {i}: {speaker.name}")
        
        except Exception as e:
            logger.error(f"Error refreshing device lists: {e}")
    
    def set_input_device(self, device_id: Union[int, str]) -> bool:
        """
        Set the input device.
        
        Args:
            device_id: Device ID or name
            
        Returns:
            True if successful, False otherwise
        """
        with self.input_device_lock:
            try:
                # Try to set the device
                if isinstance(device_id, str):
                    # Find device by name
                    for i, device in enumerate(self.input_devices):
                        if device_id.lower() in device['name'].lower():
                            device_id = i
                            break
                    else:
                        logger.warning(f"Input device '{device_id}' not found")
                        return False
                
                # Test the device
                try:
                    sd.check_input_settings(device=device_id, channels=self.channels, samplerate=self.sample_rate)
                    self.input_device = device_id
                    logger.info(f"Input device set to {device_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error setting input device {device_id}: {e}")
                    return False
            
            except Exception as e:
                logger.error(f"Error setting input device: {e}")
                return False
    
    def set_output_device(self, device_id: Union[int, str]) -> bool:
        """
        Set the output device.
        
        Args:
            device_id: Device ID or name
            
        Returns:
            True if successful, False otherwise
        """
        with self.output_device_lock:
            try:
                # Try to set the device
                if isinstance(device_id, str):
                    # Find device by name
                    for i, device in enumerate(self.output_devices):
                        if device_id.lower() in device['name'].lower():
                            device_id = i
                            break
                    else:
                        logger.warning(f"Output device '{device_id}' not found")
                        return False
                
                # Test the device
                try:
                    sd.check_output_settings(device=device_id, channels=self.channels, samplerate=self.sample_rate)
                    self.output_device = device_id
                    logger.info(f"Output device set to {device_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error setting output device {device_id}: {e}")
                    return False
            
            except Exception as e:
                logger.error(f"Error setting output device: {e}")
                return False
    
    def play_audio(self, audio_path: str, block: bool = True, volume: float = 1.0) -> bool:
        """
        Play audio file with error handling and fallback mechanisms.
        
        Args:
            audio_path: Path to audio file
            block: Whether to block until audio playback is complete
            volume: Volume (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        with self.output_device_lock:
            # Try multiple times with exponential backoff
            for attempt in range(AUDIO_DEVICE_RETRY_COUNT):
                try:
                    # Load audio file
                    data, samplerate = sf.read(audio_path)
                    
                    # Apply volume
                    data = data * volume
                    
                    # Play audio using sounddevice
                    sd.play(data, samplerate, device=self.output_device)
                    
                    if block:
                        sd.wait()
                    
                    logger.info(f"Audio playback started: {audio_path}")
                    return True
                
                except Exception as e:
                    logger.error(f"Error playing audio (attempt {attempt+1}/{AUDIO_DEVICE_RETRY_COUNT}): {e}")
                    
                    # Try to stop any ongoing playback
                    try:
                        sd.stop()
                    except:
                        pass
                    
                    # Try alternative libraries if available
                    if attempt == 1 and SOUNDCARD_AVAILABLE:
                        try:
                            logger.info("Trying to play audio using soundcard library")
                            
                            # Get default speaker
                            speaker = sc.default_speaker()
                            
                            # Load audio file
                            data, samplerate = sf.read(audio_path)
                            
                            # Apply volume
                            data = data * volume
                            
                            # Convert to float32 if needed
                            if data.dtype != np.float32:
                                data = data.astype(np.float32)
                            
                            # Ensure stereo
                            if len(data.shape) == 1:
                                data = np.column_stack((data, data))
                            
                            # Play audio
                            speaker.play(data, samplerate=samplerate)
                            
                            logger.info(f"Audio playback started using soundcard: {audio_path}")
                            return True
                        
                        except Exception as sc_e:
                            logger.error(f"Error playing audio with soundcard: {sc_e}")
                    
                    elif attempt == 2 and MINIAUDIO_AVAILABLE:
                        try:
                            logger.info("Trying to play audio using miniaudio library")
                            
                            # Play audio
                            stream = miniaudio.stream_file(audio_path)
                            device = miniaudio.PlaybackDevice()
                            device.start(stream)
                            
                            if block:
                                while device.is_playing():
                                    time.sleep(0.1)
                            
                            logger.info(f"Audio playback started using miniaudio: {audio_path}")
                            return True
                        
                        except Exception as ma_e:
                            logger.error(f"Error playing audio with miniaudio: {ma_e}")
                    
                    # Wait before retrying
                    retry_delay = AUDIO_DEVICE_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
            
            logger.error(f"Failed to play audio after {AUDIO_DEVICE_RETRY_COUNT} attempts")
            return False
    
    def record_audio(self, duration: float, callback: Optional[Callable] = None) -> Optional[str]:
        """
        Record audio with error handling and fallback mechanisms.
        
        Args:
            duration: Recording duration in seconds
            callback: Optional callback function for real-time processing
            
        Returns:
            Path to recorded audio file, or None if failed
        """
        with self.input_device_lock:
            # Try multiple times with exponential backoff
            for attempt in range(AUDIO_DEVICE_RETRY_COUNT):
                try:
                    # Calculate number of frames
                    frames = int(duration * self.sample_rate)
                    
                    # Record audio
                    recording = sd.rec(
                        frames,
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        device=self.input_device
                    )
                    
                    # Wait for recording to complete
                    sd.wait()
                    
                    # Save to file
                    from utils import get_temp_file_path
                    output_path = get_temp_file_path("recording", ".wav")
                    sf.write(output_path, recording, self.sample_rate)
                    
                    logger.info(f"Audio recording saved to {output_path}")
                    return output_path
                
                except Exception as e:
                    logger.error(f"Error recording audio (attempt {attempt+1}/{AUDIO_DEVICE_RETRY_COUNT}): {e}")
                    
                    # Try to stop any ongoing recording
                    try:
                        sd.stop()
                    except:
                        pass
                    
                    # Try alternative libraries if available
                    if attempt == 1 and SOUNDCARD_AVAILABLE:
                        try:
                            logger.info("Trying to record audio using soundcard library")
                            
                            # Get default microphone
                            mic = sc.default_microphone()
                            
                            # Record audio
                            recording = mic.record(int(duration * self.sample_rate), self.sample_rate, self.channels)
                            
                            # Save to file
                            from utils import get_temp_file_path
                            output_path = get_temp_file_path("recording_sc", ".wav")
                            sf.write(output_path, recording, self.sample_rate)
                            
                            logger.info(f"Audio recording saved to {output_path} using soundcard")
                            return output_path
                        
                        except Exception as sc_e:
                            logger.error(f"Error recording audio with soundcard: {sc_e}")
                    
                    # Wait before retrying
                    retry_delay = AUDIO_DEVICE_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
            
            logger.error(f"Failed to record audio after {AUDIO_DEVICE_RETRY_COUNT} attempts")
            return None
    
    def test_audio_devices(self) -> Tuple[bool, bool]:
        """
        Test audio input and output devices.
        
        Returns:
            Tuple of (input_ok, output_ok)
        """
        logger.info("Testing audio devices")
        
        # Test output device
        output_ok = False
        try:
            # Generate a short beep
            duration = 0.5  # seconds
            frequency = 440  # Hz (A4)
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            beep = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Save to file
            from utils import get_temp_file_path
            beep_path = get_temp_file_path("test_beep", ".wav")
            sf.write(beep_path, beep, self.sample_rate)
            
            # Play the beep
            output_ok = self.play_audio(beep_path, block=True, volume=0.5)
            
            logger.info(f"Output device test {'passed' if output_ok else 'failed'}")
        except Exception as e:
            logger.error(f"Error testing output device: {e}")
            output_ok = False
        
        # Test input device
        input_ok = False
        try:
            # Record a short audio clip
            recording_path = self.record_audio(1.0)
            
            if recording_path:
                # Check if recording contains data
                data, _ = sf.read(recording_path)
                if len(data) > 0 and np.max(np.abs(data)) > 0.01:
                    input_ok = True
            
            logger.info(f"Input device test {'passed' if input_ok else 'failed'}")
        except Exception as e:
            logger.error(f"Error testing input device: {e}")
            input_ok = False
        
        return input_ok, output_ok
    
    def get_available_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get lists of available audio devices.
        
        Returns:
            Dictionary with 'input' and 'output' device lists
        """
        # Refresh device lists first
        self.refresh_device_lists()
        
        return {
            'input': self.input_devices,
            'output': self.output_devices
        }
