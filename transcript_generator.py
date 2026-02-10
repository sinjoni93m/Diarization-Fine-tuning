"""
Transcript Generator Helper Functions Module

Supports TWO transcript generation modes:
1. Whisper Native Mode: Uses Whisper's built-in speaker diarization — also detects language
2. Custom Diarization Mode: Uses external diarization results (WebRTC, Silero)
   and receives language hint from Whisper native detection

Language detection flow:
  Whisper Native runs first → detects language → language passed as hint to diarization methods
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
from speechbrain.pretrained import EncoderClassifier

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
    1. Whisper Native: Uses Whisper's built-in VAD and also detects language
    2. Custom Diarization: Uses external diarization results (webrtc, silero)
       with language hint from Whisper native
    """
    
    def __init__(self, audio_file_path: str, language_hint: Optional[str] = None):
        """
        Initialize the multichannel transcript generator.
        
        Args:
            audio_file_path: Path to audio file
            language_hint: Optional language hint (e.g., 'en', 'es', 'en-US').
                          For diarization mode, this is typically the language
                          detected by Whisper native.
        """
        self.audio_file_path = audio_file_path
        self.language_hint = language_hint
        self.whisper_model = None  # Lazy load
        
        # Load audio to detect channels
        self.audio, self.sr = librosa.load(audio_file_path, sr=None, mono=False)
        
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(1, -1)
        
        self.num_channels = self.audio.shape[0]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lid_model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            run_opts={"device": self.device}
        )

        self.channel_languages: Dict[int, Dict[str, Any]] = {}
        self._detect_languages_per_channel()

    def _detect_languages_per_channel(self, min_seconds: float = 5.0):
        """
        Detect language per channel using SpeechBrain ECAPA-TDNN.
        Stores result in self.channel_languages.
        """
        print("    Detecting language per channel using SpeechBrain LID...")

        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]

            # Take up to first N seconds for stability
            max_samples = int(min_seconds * self.sr)
            audio_slice = channel_audio[:max_samples]

            # Convert to torch + resample to 16k
            audio_tensor = torch.tensor(audio_slice, dtype=torch.float32)

            if self.sr != 16000:
                audio_tensor = torch.nn.functional.interpolate(
                    audio_tensor.unsqueeze(0).unsqueeze(0),
                    scale_factor=16000 / self.sr,
                    mode="linear",
                    align_corners=False
                ).squeeze()

            with torch.no_grad():
                prediction = self.lid_model.classify_batch(audio_tensor.unsqueeze(0))

            lang = prediction[3][0]          # e.g. 'en'
            score = float(prediction[2][0].max())

            # Whisper only supports ISO-639-1
            lang = self._get_base_language_code(lang)
            if not self.is_language_supported(lang):
                logger.warning(f"Channel {ch_idx}: {lang} not supported by Whisper, falling back to 'en'")
                lang = "en"

            self.channel_languages[ch_idx] = {
                "language": lang,
                "confidence": score
            }

            print(f"      Channel {ch_idx}: {lang} (confidence={score:.2f})")

    
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
        Transcribe a single speech segment with a specified language.
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
    # MODE 1: WHISPER NATIVE DIARIZATION (runs first to detect language)
    # =============================================================================
    
    def transcribe_with_whisper_native(self) -> Dict[str, Any]:
        """
        MODE 1: Transcribe using WHISPER's NATIVE speaker diarization.
        
        Also performs language detection. The detected language is stored in
        the transcript output under 'primary_language' so it can be extracted
        by the caller and passed to diarization-based methods.
        
        Returns:
            Dictionary containing transcript segments, metadata, and detected language
        """
        print("    [MODE 1: WHISPER NATIVE DIARIZATION + LANGUAGE DETECTION]")
        
        self._load_whisper_model()
        
        all_transcribed_segments = []
        detected_languages = set()
        
        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]
            
            print(f"    Transcribing channel {ch_idx} with Whisper native diarization...")
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, channel_audio, self.sr)
                    
                    transcribe_options = {
                        "verbose": False,
                        "word_timestamps": True,
                    }
                    
                    # Add language hint if user provided one explicitly
                    if self.language_hint:
                        transcribe_options["language"] = self.channel_languages[ch_idx]["language"]
                        
                    # Otherwise let Whisper auto-detect
                    
                    result = self.whisper_model.transcribe(temp_path, **transcribe_options)
                    os.unlink(temp_path)
                    
                    channel_lang = self.channel_languages[ch_idx]["language"]
                    
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
                    
                    print(f"      ✓ Transcribed {len(whisper_segments)} segments (detected: {channel_lang})")
                    
            except Exception as e:
                logger.error(f"Error transcribing channel {ch_idx} with Whisper native: {e}")
                continue
        
        all_transcribed_segments.sort(key=lambda x: (x["start"], x["end"]))
        
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
    # MODE 2: CUSTOM DIARIZATION (uses language from Whisper native)
    # =============================================================================
    
    def transcribe_with_custom_diarization(self, diarization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        MODE 2: Transcribe speech segments using CUSTOM diarization results.
        
        Uses diarization results from webrtc or silero.
        The language_hint (set during __init__) should be the language detected
        by Whisper native for consistency across all transcript methods.
        
        Args:
            diarization_data: Diarization results containing speech segments per channel
        
        Returns:
            Dictionary containing transcript segments and metadata
        """
        print("    [MODE 2: CUSTOM DIARIZATION]")
        
        # Use the language hint directly (which should come from Whisper native detection)
        language_code = self.channel_languages[ch_idx]["language"] if self.language_hint else None
        
        # If no language hint was provided (shouldn't happen in normal flow), 
        # fall back to detecting from audio
        if not language_code:
            logger.warning("No language hint provided to custom diarization mode — detecting from audio")
            language_code = self._detect_language_from_diarization(diarization_data)
        
        print(f"    Using language: {language_code}")
        
        transcribed_segments = []
        channel_data = diarization_data.get("channel_data", {})
        
        for ch_idx in range(self.num_channels):
            channel_audio = self.audio[ch_idx]
            channel_info = channel_data.get(ch_idx, {})
            speech_segments = channel_info.get("speech_segments", [])
            
            if not speech_segments:
                continue
            
            print(f"    Transcribing {len(speech_segments)} segments from channel {ch_idx} (language: {language_code})...")
            
            for segment_info in speech_segments:
                start_sample = int(segment_info["start"] * self.sr)
                end_sample = int(segment_info["end"] * self.sr)
                segment_audio = channel_audio[start_sample:end_sample]
                
                if len(segment_audio) < self.sr * 0.1:
                    continue
                
                result = self.transcribe_segment(segment_audio, segment_info, ch_idx, language_code)
                
                if result:
                    transcribed_segments.append(result)
        
        transcribed_segments.sort(key=lambda x: (x["start"], x["end"]))
        
        return {
            "transcription_mode": "custom_diarization",
            "primary_language": language_code,
            "detected_languages": [language_code],
            "language_hint_provided": self.language_hint is not None,
            "language_hint": self.language_hint,
            "language_source": "whisper_native_detection" if self.language_hint else "fallback_detection",
            "transcript_segments": transcribed_segments,
            "num_segments": len(transcribed_segments),
            "channel_languages": self.channel_languages,
            "language_source": "speechbrain_audio_lid"

        }
    
    def _detect_language_from_diarization(self, diarization_data: Dict[str, Any]) -> str:
        """
        Fallback language detection from diarization segments.
        Only used if Whisper native didn't run first (edge case).
        """
        self._load_whisper_model()
        
        channel_data = diarization_data.get("channel_data", {})
        sample_segments = []
        max_samples = 5
        
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
            return "en"
        
        try:
            ch_idx, segment_info = sample_segments[0]
            channel_audio = self.audio[ch_idx]
            
            start_sample = int(segment_info["start"] * self.sr)
            end_sample = int(segment_info["end"] * self.sr)
            segment_audio = channel_audio[start_sample:end_sample]
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, segment_audio, self.sr)
                
                audio_for_detection = whisper.load_audio(temp_path)
                audio_for_detection = whisper.pad_or_trim(audio_for_detection)
                mel = whisper.log_mel_spectrogram(audio_for_detection).to(self.whisper_model.device)
                _, probs = self.whisper_model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                
                os.unlink(temp_path)
                
                print(f"    Fallback language detection: {detected_lang}")
                return detected_lang
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"
    
    # =============================================================================
    # BACKWARDS COMPATIBILITY
    # =============================================================================
    
    def transcribe_speech_segments_from_diarization(self, diarization_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.transcribe_with_custom_diarization(diarization_data)