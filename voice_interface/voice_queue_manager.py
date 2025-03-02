"""
Voice queue manager for handling voice operations in a thread-safe manner.
"""

import time
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from cachetools import LRUCache
from diskcache import Cache

from utils import logger
from config import (
    VOICE_QUEUE_SIZE,
    VOICE_PROCESSING_THREADS,
    VOICE_RATE_LIMIT,
    VOICE_CACHE_ENABLED,
    VOICE_CACHE_SIZE
)

class VoiceQueueManager:
    """
    Manages voice operation queues and rate limiting.
    """
    
    def __init__(self):
        """Initialize the voice queue manager."""
        # Create separate queues for different types of operations
        self.tts_queue = queue.PriorityQueue(maxsize=VOICE_QUEUE_SIZE)
        self.stt_queue = queue.PriorityQueue(maxsize=VOICE_QUEUE_SIZE)
        self.audio_model_queue = queue.PriorityQueue(maxsize=VOICE_QUEUE_SIZE)
        
        # Create locks for thread safety
        self.tts_lock = threading.Lock()
        self.stt_lock = threading.Lock()
        self.audio_model_lock = threading.Lock()
        
        # Create rate limiters
        self.last_tts_time = 0
        self.last_stt_time = 0
        self.last_audio_model_time = 0
        
        # Create TTS cache
        self.tts_cache = LRUCache(maxsize=VOICE_CACHE_SIZE) if VOICE_CACHE_ENABLED else None
        self.tts_disk_cache = Cache('./temp/tts_cache') if VOICE_CACHE_ENABLED else None
        
        # Create worker threads
        self.tts_workers = []
        self.stt_workers = []
        self.audio_model_workers = []
        self.should_stop = False
        
        logger.info("VoiceQueueManager initialized")
    
    def start_workers(self):
        """Start worker threads for processing voice operations."""
        logger.info("Starting voice queue workers")
        
        # Start TTS workers
        for i in range(VOICE_PROCESSING_THREADS):
            worker = threading.Thread(
                target=self._tts_worker,
                name=f"tts-worker-{i}",
                daemon=True
            )
            worker.start()
            self.tts_workers.append(worker)
        
        # Start STT workers
        for i in range(VOICE_PROCESSING_THREADS):
            worker = threading.Thread(
                target=self._stt_worker,
                name=f"stt-worker-{i}",
                daemon=True
            )
            worker.start()
            self.stt_workers.append(worker)
        
        # Start audio model workers
        for i in range(VOICE_PROCESSING_THREADS):
            worker = threading.Thread(
                target=self._audio_model_worker,
                name=f"audio-model-worker-{i}",
                daemon=True
            )
            worker.start()
            self.audio_model_workers.append(worker)
        
        logger.info(f"Started {len(self.tts_workers) + len(self.stt_workers) + len(self.audio_model_workers)} voice queue workers")
    
    def stop_workers(self):
        """Stop all worker threads."""
        logger.info("Stopping voice queue workers")
        self.should_stop = True
        
        # Wait for workers to finish
        for worker in self.tts_workers + self.stt_workers + self.audio_model_workers:
            worker.join(timeout=2)
        
        # Clear queues
        self._clear_queue(self.tts_queue)
        self._clear_queue(self.stt_queue)
        self._clear_queue(self.audio_model_queue)
        
        # Reset worker lists
        self.tts_workers = []
        self.stt_workers = []
        self.audio_model_workers = []
        self.should_stop = False
        
        logger.info("Voice queue workers stopped")
    
    def _clear_queue(self, q):
        """Clear a queue."""
        try:
            while not q.empty():
                q.get_nowait()
                q.task_done()
        except Exception:
            pass
    
    def enqueue_tts(self, text: str, priority: int = 1, callback: Optional[Callable] = None) -> bool:
        """
        Enqueue a text-to-speech operation.
        
        Args:
            text: Text to convert to speech
            priority: Priority (lower number = higher priority)
            callback: Callback function to call with the result
            
        Returns:
            True if enqueued successfully, False otherwise
        """
        # Check if text is in cache
        if VOICE_CACHE_ENABLED and text in self.tts_cache:
            logger.info(f"TTS cache hit for text: {text[:50]}...")
            if callback:
                callback(self.tts_cache[text])
            return True
        
        # Check if text is in disk cache
        if VOICE_CACHE_ENABLED and self.tts_disk_cache is not None:
            cache_key = f"tts_{hash(text)}"
            if cache_key in self.tts_disk_cache:
                logger.info(f"TTS disk cache hit for text: {text[:50]}...")
                if callback:
                    callback(self.tts_disk_cache[cache_key])
                return True
        
        try:
            # Add to queue with priority and timestamp
            timestamp = time.time()
            self.tts_queue.put((priority, timestamp, text, callback), block=False)
            logger.info(f"Enqueued TTS operation with priority {priority}: {text[:50]}...")
            return True
        except queue.Full:
            logger.warning(f"TTS queue full, dropping operation: {text[:50]}...")
            return False
    
    def enqueue_stt(self, audio_path: str, priority: int = 1, callback: Optional[Callable] = None) -> bool:
        """
        Enqueue a speech-to-text operation.
        
        Args:
            audio_path: Path to audio file
            priority: Priority (lower number = higher priority)
            callback: Callback function to call with the result
            
        Returns:
            True if enqueued successfully, False otherwise
        """
        try:
            # Add to queue with priority and timestamp
            timestamp = time.time()
            self.stt_queue.put((priority, timestamp, audio_path, callback), block=False)
            logger.info(f"Enqueued STT operation with priority {priority}: {audio_path}")
            return True
        except queue.Full:
            logger.warning(f"STT queue full, dropping operation: {audio_path}")
            return False
    
    def enqueue_audio_model(self, audio_path: str, priority: int = 1, callback: Optional[Callable] = None) -> bool:
        """
        Enqueue an audio model operation.
        
        Args:
            audio_path: Path to audio file
            priority: Priority (lower number = higher priority)
            callback: Callback function to call with the result
            
        Returns:
            True if enqueued successfully, False otherwise
        """
        try:
            # Add to queue with priority and timestamp
            timestamp = time.time()
            self.audio_model_queue.put((priority, timestamp, audio_path, callback), block=False)
            logger.info(f"Enqueued audio model operation with priority {priority}: {audio_path}")
            return True
        except queue.Full:
            logger.warning(f"Audio model queue full, dropping operation: {audio_path}")
            return False
    
    def _tts_worker(self):
        """Worker thread for processing text-to-speech operations."""
        logger.info(f"TTS worker {threading.current_thread().name} started")
        
        while not self.should_stop:
            try:
                # Get item from queue with timeout
                try:
                    priority, timestamp, text, callback = self.tts_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Apply rate limiting
                self._apply_rate_limit("tts")
                
                # Process TTS operation (actual implementation will be in OpenAIVoiceInterface)
                # This is just a placeholder - the actual TTS call will be made by the callback
                if callback:
                    try:
                        callback(text)
                    except Exception as e:
                        logger.error(f"Error in TTS callback: {e}")
                
                # Mark task as done
                self.tts_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in TTS worker: {e}")
                time.sleep(1)  # Prevent tight loop on error
        
        logger.info(f"TTS worker {threading.current_thread().name} stopped")
    
    def _stt_worker(self):
        """Worker thread for processing speech-to-text operations."""
        logger.info(f"STT worker {threading.current_thread().name} started")
        
        while not self.should_stop:
            try:
                # Get item from queue with timeout
                try:
                    priority, timestamp, audio_path, callback = self.stt_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Apply rate limiting
                self._apply_rate_limit("stt")
                
                # Process STT operation (actual implementation will be in OpenAIVoiceInterface)
                # This is just a placeholder - the actual STT call will be made by the callback
                if callback:
                    try:
                        callback(audio_path)
                    except Exception as e:
                        logger.error(f"Error in STT callback: {e}")
                
                # Mark task as done
                self.stt_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in STT worker: {e}")
                time.sleep(1)  # Prevent tight loop on error
        
        logger.info(f"STT worker {threading.current_thread().name} stopped")
    
    def _audio_model_worker(self):
        """Worker thread for processing audio model operations."""
        logger.info(f"Audio model worker {threading.current_thread().name} started")
        
        while not self.should_stop:
            try:
                # Get item from queue with timeout
                try:
                    priority, timestamp, audio_path, callback = self.audio_model_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Apply rate limiting
                self._apply_rate_limit("audio_model")
                
                # Process audio model operation (actual implementation will be in OpenAIVoiceInterface)
                # This is just a placeholder - the actual audio model call will be made by the callback
                if callback:
                    try:
                        callback(audio_path)
                    except Exception as e:
                        logger.error(f"Error in audio model callback: {e}")
                
                # Mark task as done
                self.audio_model_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in audio model worker: {e}")
                time.sleep(1)  # Prevent tight loop on error
        
        logger.info(f"Audio model worker {threading.current_thread().name} stopped")
    
    def _apply_rate_limit(self, operation_type: str):
        """
        Apply rate limiting to an operation.
        
        Args:
            operation_type: Type of operation ("tts", "stt", or "audio_model")
        """
        if operation_type == "tts":
            with self.tts_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_tts_time
                if time_since_last < 1.0 / VOICE_RATE_LIMIT:
                    sleep_time = 1.0 / VOICE_RATE_LIMIT - time_since_last
                    time.sleep(sleep_time)
                self.last_tts_time = time.time()
        
        elif operation_type == "stt":
            with self.stt_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_stt_time
                if time_since_last < 1.0 / VOICE_RATE_LIMIT:
                    sleep_time = 1.0 / VOICE_RATE_LIMIT - time_since_last
                    time.sleep(sleep_time)
                self.last_stt_time = time.time()
        
        elif operation_type == "audio_model":
            with self.audio_model_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_audio_model_time
                if time_since_last < 1.0 / VOICE_RATE_LIMIT:
                    sleep_time = 1.0 / VOICE_RATE_LIMIT - time_since_last
                    time.sleep(sleep_time)
                self.last_audio_model_time = time.time()
    
    def add_to_tts_cache(self, text: str, audio_path: str):
        """
        Add a TTS result to the cache.
        
        Args:
            text: Text that was converted to speech
            audio_path: Path to the generated audio file
        """
        if VOICE_CACHE_ENABLED and self.tts_cache is not None:
            self.tts_cache[text] = audio_path
            
            # Also add to disk cache
            if self.tts_disk_cache is not None:
                cache_key = f"tts_{hash(text)}"
                self.tts_disk_cache[cache_key] = audio_path
    
    def clear_tts_cache(self):
        """Clear the TTS cache."""
        if VOICE_CACHE_ENABLED:
            if self.tts_cache is not None:
                self.tts_cache.clear()
            if self.tts_disk_cache is not None:
                self.tts_disk_cache.clear()
            logger.info("TTS cache cleared")
