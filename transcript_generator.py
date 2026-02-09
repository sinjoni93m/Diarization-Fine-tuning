"""
Transcript Generator Helper Functions Module

Supports TWO transcript generation modes:
1. Custom Diarization Mode: Uses external diarization results (from our 4 methods)
2. Whisper Native Mode: Uses Whisper's built-in speaker diarization
"""

import os
import tempfile
import warnings
import numpy as np
import librosa
import soundfile as sf
import whisper
import logging
from typing import Optional, Dict, Any, List

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Whisper supported languages (base ISO 639-1 codes)
WHISPER_SUPPORTED_LANGUAGES = {
    'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl',
    'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro',
    'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy',
    'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu',
    'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km',
    'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo',
    'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg',
    'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'
}


class MultichannelTranscriptGenerator:
    """
    Generate transcripts for multichannel audio files.
    
    Supports two modes:
    1. Custom Diarization: Uses external diarization results (energy, spectral, webrtc, silero)
    2. Whisper Native: Uses Whisper's built-in speaker diarization
    """
    
    def __init__(self, audio_file_path: str, language_hint: Optional[str] = None):
        """
        Initialize the multichannel transcript generator.
        
        Args:
            audio_file_path: Path to audio file
            language_hint: Optional language hint (e.g., 'en', 'es', 'en-US')
        """
        self.audio_file_path = audio_file_path
        self.language_hint = language_hint
        self.whisper_model = None  # Lazy load
        self.detected_language = None  # Cache detected language
        
        # Load audio to detect channels
        self.audio, self.sr = librosa.load(audio_file_path, sr=None, mono=False)
        
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(1, -1)
        
        self.num_channels = self.audio.shape[0]
    
    def _load_whisper_model(self):
        """
        Lazy load Whisper model (large-v3 for best accuracy).
        """
        if self.whisper_model is None:
            print("    Loading Whisper model (large-v3)...")
            self.whisper_model = whisper.load_model("large-v3")
    
    def _get_base_language_code(self, language: Optional[str]) -> Optional[str]:
        """
        Extract base language code from dialect specification.
        
        Examples:
            'en-US' -> 'en'
            'es-MX' -> 'es'
            'en' -> 'en'
        """
        if not language:
            return None
        
        # If already a base code (2 letters), return as-is
        if len(language) == 2:
            return language.lower()
        
        # Extract base language from dialect code
        base_lang = language.split('-')[0].lower()
        return base_lang
    
    def is_language_supported(self, language: Optional[str]) -> bool:
        """
        Check if language is supported by Whisper.
        
        Args:
            language: Language code (e.g., 'en', 'es-MX')
        
        Returns:
            True if supported, False otherwise
        """
        if not language:
            return True  # Auto-detect is always supported
        
        base_lang = self._get_base_language_code(language)
        return base_lang in WHISPER_SUPPORTED_LANGUAGES
    
    def detect_language_from_audio(self, diarization_data: Dict[str, Any]) -> str:
        """
        Detect language once from a sample of the audio for consistent transcription.
        Uses the first few speech segments to determine the language.
        
        Args:
            diarization_data: Diarization results containing speech segments
        
        Returns:
            Detected language code (e.g., 'en', 'es')
        """
        # If language hint provided, use it
        if self.language_hint:
            return self._get_base_language_code(self.language_hint)
        
        # If already detected, return cached result
        if self.detected_language:
            return self.detected_language
        
        # Lazy load model
        self._load_whisper_model()
        
        # Collect first few speech segments for language detection
        channel_data = diarization_data.get("channel_data", {})
        sample_segments = []
        max_samples = 5  # Use first 5 segments for detection
        
        for ch_idx in range(self.num_channels):
            channel_info = channel_data.get(ch_idx, {})
            speech_segments = channel_info.get("speech_segments", [])
            
            for segment in speech_segments[:max_samples]:
                sample_segments.append((ch_idx, segment))
                if len(sample_segments) >= max_samples:
                    break
            
            if len(sample_segments) >= max_samples:
                break
        
        if not sample_segments:
            logger.warning("No speech segments found for language detection")
            return "en"  # Default fallback
        
        # Extract audio from sample segments and detect language
        try:
            # Use first segment for language detection
            ch_idx, segment_info = sample_segments[0]
            channel_audio = self.audio[ch_idx]
            
            start_sample = int(segment_info["start"] * self.sr)
            end_sample = int(segment_info["end"] * self.sr)
            segment_audio = channel_audio[start_sample:end_sample]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, segment_audio, self.sr)
                
                # Detect language using Whisper
                audio_for_detection = whisper.load_audio(temp_path)
                audio_for_detection = whisper.pad_or_trim(audio_for_detection)
                mel = whisper.log_mel_spectrogram(audio_for_detection).to(self.whisper_model.device)
                _, probs = self.whisper_model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                
                # Clean up
                os.unlink(temp_path)
                
                self.detected_language = detected_lang
                print(f"    Detected language: {detected_lang}")
                
                return detected_lang
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"  # Default fallback
    
    def transcribe_segment(self, audio_segment: np.ndarray, segment_info: Dict[str, Any],
                          channel_idx: int, language_code: str) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single speech segment with a specified language.
        
        Args:
            audio_segment: Audio data for the segment
            segment_info: Segment metadata (start, end times)
            channel_idx: Channel index
            language_code: Language to use for transcription
        
        Returns:
            Dictionary with transcription results or None on error
        """
        try:
            # Lazy load model
            self._load_whisper_model()
            
            # Create temporary file for segment
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write segment audio to temporary file
                sf.write(temp_path, audio_segment, self.sr)
                
                # Transcribe using Whisper with specified language
                transcribe_options = {
                    "verbose": False,
                    "language": language_code  # Use detected/hint language for all segments
                }
                
                result = self.whisper_model.transcribe(temp_path, **transcribe_options)
                
                # Clean up temp file
                os.unlink(temp_path)
                
                return {
                    "start": segment_info["start"],
                    "end": segment_info["end"],
                    "speaker": f"channel_{channel_idx}",
                    "detected_language": language_code,
                    "text": result["text"].strip()
                }
                
        except Exception as e:
            logger.error(f"Error transcribing segment at {segment_info['start']}-{segment_info['end']}: {e}")
            return None
    
    # =============================================================================
    # MODE 1: CUSTOM DIARIZATION (Uses external diarization results)
    # =============================================================================
    
    def transcribe_with_custom_diarization(self, diarization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        MODE 1: Transcribe speech segments using CUSTOM diarization results.
        
        Uses diarization results from one of our methods (energy, spectral, webrtc, silero).
        Detects language once and uses it for all segments for consistency.
        
        Args:
            diarization_data: Diarization results containing speech segments per channel
        
        Returns:
            Dictionary containing transcript segments and metadata
        """
        print("    [MODE 1: CUSTOM DIARIZATION]")
        
        # Detect language once for the entire audio
        detected_language = self.detect_language_from_audio(diarization_data)
        
        transcribed_segments = []
        channel_data = diarization_data.get("channel_data", {})
        
        # Process each channel
        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]
            channel_info = channel_data.get(ch_idx, {})
            speech_segments = channel_info.get("speech_segments", [])
            
            if not speech_segments:
                continue
            
            print(f"    Transcribing {len(speech_segments)} segments from channel {ch_idx} (language: {detected_language})...")
            
            # Transcribe each speech segment with the detected language
            for segment_info in speech_segments:
                # Extract audio for this segment
                start_sample = int(segment_info["start"] * self.sr)
                end_sample = int(segment_info["end"] * self.sr)
                segment_audio = channel_audio[start_sample:end_sample]
                
                # Skip very short segments
                if len(segment_audio) < self.sr * 0.1:  # Less than 0.1 seconds
                    continue
                
                # Transcribe segment with detected language
                result = self.transcribe_segment(segment_audio, segment_info, ch_idx, detected_language)
                
                if result:
                    transcribed_segments.append(result)
        
        # Sort all segments by start time, then end time
        transcribed_segments.sort(key=lambda x: (x["start"], x["end"]))
        
        return {
            "transcription_mode": "custom_diarization",
            "primary_language": detected_language,
            "detected_languages": [detected_language],
            "language_hint_provided": self.language_hint is not None,
            "language_hint": self.language_hint,
            "transcript_segments": transcribed_segments,
            "num_segments": len(transcribed_segments)
        }
    
    # =============================================================================
    # MODE 2: WHISPER NATIVE DIARIZATION (Uses Whisper's built-in VAD)
    # =============================================================================
    
    def transcribe_with_whisper_native(self) -> Dict[str, Any]:
        """
        MODE 2: Transcribe using WHISPER's NATIVE speaker diarization.
        
        Lets Whisper handle both speech detection and transcription using its
        built-in VAD (Voice Activity Detection). This is simpler but gives less
        control over diarization parameters.
        
        Returns:
            Dictionary containing transcript segments and metadata
        """
        print("    [MODE 2: WHISPER NATIVE DIARIZATION]")
        
        # Lazy load model
        self._load_whisper_model()
        
        all_transcribed_segments = []
        detected_languages = set()
        
        # Process each channel separately
        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]
            
            print(f"    Transcribing channel {ch_idx} with Whisper native diarization...")
            
            try:
                # Create temporary file for entire channel
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, channel_audio, self.sr)
                    
                    # Transcribe options
                    transcribe_options = {
                        "verbose": False,
                        "word_timestamps": True,  # Enable word-level timestamps
                    }
                    
                    # Add language hint if provided
                    if self.language_hint:
                        transcribe_options["language"] = self._get_base_language_code(self.language_hint)
                    
                    # Transcribe using Whisper's native VAD and segmentation
                    result = self.whisper_model.transcribe(temp_path, **transcribe_options)
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    
                    # Extract detected language
                    detected_lang = result.get("language", "en")
                    detected_languages.add(detected_lang)
                    
                    # Process segments from Whisper's output
                    whisper_segments = result.get("segments", [])
                    
                    for segment in whisper_segments:
                        # Whisper provides automatic segmentation with timestamps
                        all_transcribed_segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "speaker": f"channel_{ch_idx}",
                            "detected_language": detected_lang,
                            "text": segment["text"].strip(),
                            "confidence": segment.get("no_speech_prob", None)  # Whisper's confidence
                        })
                    
                    print(f"      ✓ Transcribed {len(whisper_segments)} segments")
                    
            except Exception as e:
                logger.error(f"Error transcribing channel {ch_idx} with Whisper native: {e}")
                continue
        
        # Sort all segments by start time, then end time
        all_transcribed_segments.sort(key=lambda x: (x["start"], x["end"]))
        
        # Determine primary language (most common)
        primary_language = max(detected_languages, key=lambda lang: 
                             sum(1 for seg in all_transcribed_segments if seg["detected_language"] == lang)
                             ) if detected_languages else "en"
        
        return {
            "transcription_mode": "whisper_native",
            "primary_language": primary_language,
            "detected_languages": sorted(list(detected_languages)),
            "language_hint_provided": self.language_hint is not None,
            "language_hint": self.language_hint,
            "transcript_segments": all_transcribed_segments,
            "num_segments": len(all_transcribed_segments)
        }
    
    # =============================================================================
    # CONVENIENCE METHODS (Backwards compatibility)
    # =============================================================================
    
    def transcribe_speech_segments_from_diarization(self, diarization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backwards compatibility wrapper for custom diarization mode.
        
        This is the original method name - now redirects to transcribe_with_custom_diarization.
        """
        return self.transcribe_with_custom_diarization(diarization_data)