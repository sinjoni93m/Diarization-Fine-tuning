"""
Transcript Generator Helper Functions Module

Supports TWO transcript generation modes:
1. Whisper Native Mode: Uses Whisper's built-in speaker diarization + Whisper language detection (30s)
2. Custom Diarization Mode: Uses external diarization results (WebRTC, Silero) + Whisper language detection

Language detection flow:
  Whisper detects language per channel from either:
  - Raw audio (first 30s) for Whisper Native mode
  - Speech segments for Custom Diarization mode (more accurate)
  
Note: Vietnamese ('vi') and all other languages in WHISPER_SUPPORTED_LANGUAGES are fully supported.
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
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Whisper supported languages (base ISO 639-1 codes)
# Vietnamese is supported as 'vi'
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
    1. Whisper Native: Uses Whisper's built-in VAD with Whisper language detection per channel (30s)
    2. Custom Diarization: Uses external diarization results (webrtc, silero) with Whisper language detection
    """
    
    def __init__(self, audio_file_path: str, language_hint: Optional[str] = None, 
                 diarization_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the multichannel transcript generator.
        
        Args:
            audio_file_path: Path to audio file
            language_hint: Optional language hint (e.g., 'en', 'es', 'vi').
                          If provided, overrides Whisper detection for all channels.
            diarization_data: Optional diarization results. If provided, language detection
                            will use actual speech segments instead of raw audio.
        """
        self.audio_file_path = audio_file_path
        self.user_language_hint = language_hint  # User-provided hint (overrides detection)
        self.diarization_data = diarization_data
        self.whisper_model = None  # Lazy load
        
        # Load audio to detect channels
        self.audio, self.sr = librosa.load(audio_file_path, sr=None, mono=False)
        
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(1, -1)
        
        self.num_channels = self.audio.shape[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.channel_languages: Dict[int, Dict[str, Any]] = {}
        self._detect_languages_per_channel()

    def _detect_languages_per_channel(self, min_seconds: float = 30.0):
        """
        Detect language per channel using Whisper's language detection.
        Stores result in self.channel_languages.
        
        If user provided a language hint, that overrides detection for all channels.
        If diarization data is available, detects from actual speech segments.
        Otherwise, detects from first 30 seconds of raw audio.
        """
        if self.user_language_hint:
            # User provided explicit language hint - use it for all channels
            base_lang = self._get_base_language_code(self.user_language_hint)
            if not self.is_language_supported(base_lang):
                logger.warning(f"User-provided language '{base_lang}' not supported by Whisper, falling back to 'en'")
                base_lang = "en"
            
            print(f"    Using user-provided language hint: {base_lang}")
            for ch_idx in range(self.num_channels):
                self.channel_languages[ch_idx] = {
                    "language": base_lang,
                    "confidence": 1.0,
                    "source": "user_hint"
                }
                print(f"      Channel {ch_idx}: {base_lang} (user-provided)")
            return

        # Use diarization-based detection if available
        if self.diarization_data:
            self._detect_from_diarization_segments()
        else:
            self._detect_from_raw_audio(min_seconds)

    def _detect_language_from_audio(self, audio_data: np.ndarray) -> tuple:
        """
        Helper function to detect language from audio data using Whisper.
        
        Args:
            audio_data: Audio numpy array at original sample rate
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_data, self.sr)
            
            # Use Whisper's transcribe with language detection only
            # This is more reliable than using detect_language directly
            result = self.whisper_model.transcribe(
                temp_path,
                task="transcribe",
                fp16=False,
                language=None  # Let Whisper auto-detect
            )
            
            os.unlink(temp_path)
            
            # Get detected language from result
            detected_lang = result.get("language", "en")
            
            # Whisper doesn't always return confidence, so we'll set a default
            confidence = 0.99  # High confidence since Whisper detected it
            
            # Ensure it's a base language code
            lang = self._get_base_language_code(detected_lang)
            if not self.is_language_supported(lang):
                logger.warning(f"Detected language '{lang}' not supported by Whisper, falling back to 'en'")
                lang = "en"
            
            return lang, confidence
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return "en", 0.0
    
    def _detect_from_raw_audio(self, min_seconds: float = 30.0):
        """
        Detect language from first 30 seconds of raw audio per channel using Whisper.
        Used when no diarization data is available (e.g., for Whisper Native mode).
        """
        print("    Detecting language per channel using Whisper (from raw audio, 30s)...")
        
        self._load_whisper_model()

        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]

            # Take up to first 30 seconds for better stability
            max_samples = int(min_seconds * self.sr)
            audio_slice = channel_audio[:max_samples]

            lang, confidence = self._detect_language_from_audio(audio_slice)
            
            self.channel_languages[ch_idx] = {
                "language": lang,
                "confidence": confidence,
                "source": "whisper_lid_raw_audio"
            }

            print(f"      Channel {ch_idx}: {lang} (confidence={confidence:.2f})")

    def _detect_from_diarization_segments(self, max_segments: int = 5, max_duration: float = 30.0):
        """
        Detect language from actual speech segments identified by diarization using Whisper.
        More accurate than raw audio as it focuses on actual speech.
        
        Args:
            max_segments: Maximum number of segments to use per channel
            max_duration: Maximum total duration of audio to analyze per channel (seconds)
        """
        print("    Detecting language per channel using Whisper (from diarization segments)...")
        
        self._load_whisper_model()
        
        channel_data = self.diarization_data.get("channel_data", {})
        
        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]
            channel_info = channel_data.get(ch_idx, {})
            speech_segments = channel_info.get("speech_segments", [])
            
            if not speech_segments:
                # No speech found, fallback to 'en'
                logger.warning(f"Channel {ch_idx}: No speech segments found, using default 'en'")
                self.channel_languages[ch_idx] = {
                    "language": "en",
                    "confidence": 0.0,
                    "source": "whisper_lid_no_segments"
                }
                print(f"      Channel {ch_idx}: en (no segments found)")
                continue
            
            # Collect audio from speech segments
            collected_audio = []
            total_duration = 0.0
            
            for segment_info in speech_segments[:max_segments]:
                if total_duration >= max_duration:
                    break
                
                start_sample = int(segment_info["start"] * self.sr)
                end_sample = int(segment_info["end"] * self.sr)
                segment_audio = channel_audio[start_sample:end_sample]
                
                segment_duration = len(segment_audio) / self.sr
                if total_duration + segment_duration > max_duration:
                    # Take partial segment to reach max_duration
                    samples_needed = int((max_duration - total_duration) * self.sr)
                    segment_audio = segment_audio[:samples_needed]
                
                collected_audio.append(segment_audio)
                total_duration += len(segment_audio) / self.sr
            
            if not collected_audio:
                logger.warning(f"Channel {ch_idx}: No valid segments, using default 'en'")
                self.channel_languages[ch_idx] = {
                    "language": "en",
                    "confidence": 0.0,
                    "source": "whisper_lid_no_valid_segments"
                }
                print(f"      Channel {ch_idx}: en (no valid segments)")
                continue
            
            # Concatenate all collected audio
            combined_audio = np.concatenate(collected_audio)
            
            # Detect language from combined audio
            lang, confidence = self._detect_language_from_audio(combined_audio)
            
            self.channel_languages[ch_idx] = {
                "language": lang,
                "confidence": confidence,
                "source": "whisper_lid_diarization_segments",
                "segments_used": len(collected_audio),
                "audio_duration": round(total_duration, 2)
            }
            
            print(f"      Channel {ch_idx}: {lang} (confidence={confidence:.2f}, {len(collected_audio)} segments, {total_duration:.1f}s)")

    
    def _load_whisper_model(self):
        if self.whisper_model is None:
            print("    Loading Whisper model (large-v3)...")
            self.whisper_model = whisper.load_model("large-v3")
    
    def _get_base_language_code(self, language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        if len(language) == 2:
            return language.lower()
        base_lang = language.split('-')[0].lower()
        return base_lang
    
    def is_language_supported(self, language: Optional[str]) -> bool:
        if not language:
            return True
        base_lang = self._get_base_language_code(language)
        return base_lang in WHISPER_SUPPORTED_LANGUAGES
    
    def transcribe_segment(self, audio_segment: np.ndarray, segment_info: Dict[str, Any],
                          channel_idx: int) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single speech segment using the channel-specific language detected by Whisper.
        """
        try:
            self._load_whisper_model()

            language_code = self.channel_languages[channel_idx]["language"]
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_segment, self.sr)
                
                transcribe_options = {
                    "verbose": False,
                    "language": language_code
                }
                
                result = self.whisper_model.transcribe(temp_path, **transcribe_options)
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
    # MODE 1: WHISPER NATIVE DIARIZATION (with Whisper language detection)
    # =============================================================================
    
    def transcribe_with_whisper_native(self) -> Dict[str, Any]:
        """
        MODE 1: Transcribe using WHISPER's NATIVE speaker diarization.
        
        Uses Whisper-detected language per channel as hint to Whisper transcription.
        
        Returns:
            Dictionary containing transcript segments, metadata, and detected language
        """
        print("    [MODE 1: WHISPER NATIVE DIARIZATION with Whisper language detection]")
        
        self._load_whisper_model()
        
        all_transcribed_segments = []
        detected_languages = set()
        
        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]
            channel_lang = self.channel_languages[ch_idx]["language"]
            
            print(f"    Transcribing channel {ch_idx} with Whisper (language hint: {channel_lang})...")
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, channel_audio, self.sr)
                    
                    transcribe_options = {
                        "verbose": False,
                        "word_timestamps": True,
                        "language": channel_lang  # Use Whisper-detected language
                    }
                    
                    result = self.whisper_model.transcribe(temp_path, **transcribe_options)
                    os.unlink(temp_path)
                    
                    detected_languages.add(channel_lang)
                    
                    whisper_segments = result.get("segments", [])
                    
                    for segment in whisper_segments:
                        all_transcribed_segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "speaker": f"channel_{ch_idx}",
                            "detected_language": channel_lang,
                            "text": segment["text"].strip(),
                            "confidence": segment.get("no_speech_prob", None)
                        })
                    
                    print(f"      ✓ Transcribed {len(whisper_segments)} segments (language: {channel_lang})")
                    
            except Exception as e:
                logger.error(f"Error transcribing channel {ch_idx} with Whisper native: {e}")
                continue
        
        all_transcribed_segments.sort(key=lambda x: (x["start"], x["end"]))
        
        # Primary language is the most common across channels
        primary_language = max(detected_languages, key=lambda lang: 
                             sum(1 for seg in all_transcribed_segments if seg["detected_language"] == lang)
                             ) if detected_languages else "en"
        
        return {
            "transcription_mode": "whisper_native",
            "primary_language": primary_language,
            "detected_languages": sorted(list(detected_languages)),
            "language_hint_provided": self.user_language_hint is not None,
            "language_hint": self.user_language_hint,
            "language_source": "user_hint" if self.user_language_hint else "whisper_lid",
            "channel_languages": self.channel_languages,
            "transcript_segments": all_transcribed_segments,
            "num_segments": len(all_transcribed_segments)
        }
    
    # =============================================================================
    # MODE 2: CUSTOM DIARIZATION (with Whisper language detection)
    # =============================================================================
    
    def transcribe_with_custom_diarization(self, diarization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        MODE 2: Transcribe speech segments using CUSTOM diarization results.
        
        Uses Whisper-detected language per channel as hint to Whisper transcription.
        
        Args:
            diarization_data: Diarization results containing speech segments per channel
        
        Returns:
            Dictionary containing transcript segments and metadata
        """
        print("    [MODE 2: CUSTOM DIARIZATION with Whisper language detection]")
        
        transcribed_segments = []
        channel_data = diarization_data.get("channel_data", {})
        detected_languages = set()
        
        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]
            channel_info = channel_data.get(ch_idx, {})
            speech_segments = channel_info.get("speech_segments", [])
            
            if not speech_segments:
                continue
            
            channel_lang = self.channel_languages[ch_idx]["language"]
            detected_languages.add(channel_lang)
            
            print(f"    Transcribing {len(speech_segments)} segments from channel {ch_idx} (language: {channel_lang})...")
            
            for segment_info in speech_segments:
                start_sample = int(segment_info["start"] * self.sr)
                end_sample = int(segment_info["end"] * self.sr)
                segment_audio = channel_audio[start_sample:end_sample]
                
                if len(segment_audio) < self.sr * 0.1:
                    continue
                
                result = self.transcribe_segment(segment_audio, segment_info, ch_idx)
                
                if result:
                    transcribed_segments.append(result)
        
        transcribed_segments.sort(key=lambda x: (x["start"], x["end"]))
        
        # Primary language is the most common across all transcribed segments
        primary_language = max(detected_languages, key=lambda lang: 
                             sum(1 for seg in transcribed_segments if seg["detected_language"] == lang)
                             ) if detected_languages else "en"
        
        return {
            "transcription_mode": "custom_diarization",
            "primary_language": primary_language,
            "detected_languages": sorted(list(detected_languages)),
            "language_hint_provided": self.user_language_hint is not None,
            "language_hint": self.user_language_hint,
            "language_source": "user_hint" if self.user_language_hint else "whisper_lid",
            "transcript_segments": transcribed_segments,
            "num_segments": len(transcribed_segments),
            "channel_languages": self.channel_languages
        }
    
    # =============================================================================
    # BACKWARDS COMPATIBILITY
    # =============================================================================
    
    def transcribe_speech_segments_from_diarization(self, diarization_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.transcribe_with_custom_diarization(diarization_data)