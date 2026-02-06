"""
Audio Diarization Helper Functions Module

Contains the AudioDiarization class with helper functions for speech/silence detection
using robust spectral feature analysis instead of simple energy thresholding.

=============================================================================
SPEECH/SILENCE DETECTION METHODOLOGY
=============================================================================

This implementation uses spectral features rather than raw energy to detect
speech segments, making it more robust to background noise.

METHOD: Spectral Feature Analysis
----------------------------------
Instead of measuring raw signal energy (which is sensitive to any noise),
we analyze the spectral characteristics of the audio:

1. SPECTRAL CENTROID
   - Measures the "center of mass" of the frequency spectrum
   - Speech has higher spectral centroids than most background noise
   - Typical range: 1000-4000 Hz for speech, lower for rumble/hum
   
2. SPECTRAL ROLLOFF
   - Frequency below which 85% of spectral energy is contained
   - Speech has characteristic rolloff patterns
   - Helps distinguish voice from white noise or machinery

THRESHOLD DETERMINATION
-----------------------
We use ADAPTIVE THRESHOLDING rather than fixed dB values:

- Fixed Threshold Problem: A -40dB threshold works in quiet rooms but
  fails in noisy environments or with different recording levels
  
- Adaptive Solution: Calculate threshold based on the audio's own
  statistical distribution using percentiles
  
- Default: 30th percentile
  * Assumes ~30% of audio is silence/noise
  * Can be adjusted: lower percentile = more sensitive (detects more speech)
                     higher percentile = less sensitive (stricter detection)

NORMALIZATION
-------------
Features are z-score normalized: (value - mean) / std_dev
This makes the detection independent of:
- Recording volume/gain
- Microphone sensitivity
- Distance from speaker

PARAMETERS TO TUNE
------------------
If detection is still problematic, adjust these parameters:

1. min_speech_duration (default: 0.3 seconds)
   - Increase to filter out brief noises
   - Decrease to capture short utterances
   
2. min_silence_duration (default: 0.3 seconds)
   - Increase to ignore brief pauses in speech
   - Decrease to detect shorter silence gaps
   
3. percentile_threshold (default: 30)
   - Increase (40-50) for noisier environments
   - Decrease (20-25) for cleaner audio
   
4. hop_length (default: 512 samples)
   - Smaller = more temporal precision, slower processing
   - Larger = faster processing, less precision

TYPICAL NOISE SCENARIOS
-----------------------
- Office environment: percentile_threshold=35
- Street/outdoor: percentile_threshold=45
- Studio/quiet room: percentile_threshold=25
- Phone/compressed audio: percentile_threshold=40

=============================================================================
"""

import librosa
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")


class AudioDiarization:

    def __init__(self, audio_path):
        self.file_path = audio_path
        self.audio, self.sr = librosa.load(audio_path, sr=None, mono=False)
        
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(1, -1)
        
        self.num_channels = self.audio.shape[0]

    def __detect_speech_robust(self, audio_channel, sr, 
                               min_speech_duration=0.3, 
                               percentile_threshold=30):
        """
        Detect speech segments using spectral features (noise-robust method).
        """
        hop_length = 512
        
        # Calculate spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_channel, sr=sr, hop_length=hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_channel, sr=sr, hop_length=hop_length, roll_percent=0.85
        )[0]
        
        # Z-score normalization to make detection independent of recording level
        centroid_norm = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-10)
        rolloff_norm = (spectral_rolloff - np.mean(spectral_rolloff)) / (np.std(spectral_rolloff) + 1e-10)
        
        # Combined score: average of normalized features
        combined_score = (centroid_norm + rolloff_norm) / 2
        
        # Adaptive threshold based on the audio's own distribution
        threshold = np.percentile(combined_score, percentile_threshold)
        
        # Determine speech frames
        speech_frames = combined_score > threshold
        
        # Convert frames to time segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            t = i * hop_length / sr
            
            if is_speech and not in_speech:
                # Start of speech segment
                speech_start = t
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                speech_duration = t - speech_start
                if speech_duration >= min_speech_duration:
                    speech_segments.append({
                        "start": round(speech_start, 3),
                        "end": round(t, 3)
                    })
                in_speech = False
        
        # Handle case where speech continues to end of audio
        if in_speech:
            t = len(audio_channel) / sr
            speech_duration = t - speech_start
            if speech_duration >= min_speech_duration:
                speech_segments.append({
                    "start": round(speech_start, 3),
                    "end": round(t, 3)
                })
        
        return speech_segments

    def __detect_silence_robust(self, audio_channel, sr, 
                                min_silence_duration=0.3,
                                percentile_threshold=30):
        """
        Detect silence segments using spectral features (noise-robust method).
        
        This is the inverse of speech detection - regions with low spectral
        activity are classified as silence.
        
        Args:
            audio_channel: Audio signal array
            sr: Sample rate
            min_silence_duration: Minimum duration (seconds) to consider as silence
            percentile_threshold: Percentile for adaptive threshold
        
        Returns:
            List of silence segment dictionaries with 'start', 'end', and 'duration'
        """
        hop_length = 512
        
        # Calculate spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_channel, sr=sr, hop_length=hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_channel, sr=sr, hop_length=hop_length, roll_percent=0.85
        )[0]
        
        # Z-score normalization
        centroid_norm = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-10)
        rolloff_norm = (spectral_rolloff - np.mean(spectral_rolloff)) / (np.std(spectral_rolloff) + 1e-10)
        
        # Combined score
        combined_score = (centroid_norm + rolloff_norm) / 2
        
        # Adaptive threshold
        threshold = np.percentile(combined_score, percentile_threshold)
        
        # Silence is where score is below threshold
        silence_frames = combined_score <= threshold
        
        # Convert frames to time segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            t = i * hop_length / sr
            
            if is_silent and not in_silence:
                # Start of silence segment
                silence_start = t
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence segment
                silence_duration = t - silence_start
                if silence_duration >= min_silence_duration:
                    silence_segments.append({
                        "start": round(silence_start, 3),
                        "end": round(t, 3),
                        "duration": round(silence_duration, 3)
                    })
                in_silence = False
        
        # Handle case where silence continues to end of audio
        if in_silence:
            t = len(audio_channel) / sr
            silence_duration = t - silence_start
            if silence_duration >= min_silence_duration:
                silence_segments.append({
                    "start": round(silence_start, 3),
                    "end": round(t, 3),
                    "duration": round(silence_duration, 3)
                })
        
        return silence_segments

    def calculate_silence_types(self, speech_segments: List[Dict], 
                               silence_segments: List[Dict], 
                               total_duration_sec: float) -> Dict:
        """
        Calculate leading, trailing, and middle silences from speech and silence segments.
        """
        if not speech_segments:
            # No speech detected - entire audio is silence
            return {
                "leading_silence_sec": round(total_duration_sec, 3),
                "leading_silence_segments": silence_segments,
                "trailing_silence_sec": 0.0,
                "trailing_silence_segments": [],
                "middle_silences": [],
                "middle_silence_total_sec": 0.0,
                "middle_silence_count": 0
            }
        
        # Sort segments by start time
        speech_segments_sorted = sorted(speech_segments, key=lambda x: x["start"])
        silence_segments_sorted = sorted(silence_segments, key=lambda x: x["start"])
        
        # Get first and last speech timestamps
        first_speech_start = speech_segments_sorted[0]["start"]
        last_speech_end = speech_segments_sorted[-1]["end"]
        
        # Categorize silences
        leading_silence_segments = []
        trailing_silence_segments = []
        middle_silences = []
        
        leading_silence_total = 0.0
        trailing_silence_total = 0.0
        middle_silence_total = 0.0
        
        for silence_seg in silence_segments_sorted:
            silence_start = silence_seg["start"]
            silence_end = silence_seg["end"]
            
            # Leading silence (before first speech)
            if silence_end <= first_speech_start:
                leading_silence_segments.append(silence_seg)
                leading_silence_total += silence_seg["duration"]
            
            # Trailing silence (after last speech)
            elif silence_start >= last_speech_end:
                trailing_silence_segments.append(silence_seg)
                trailing_silence_total += silence_seg["duration"]
            
            # Middle silence (between first and last speech)
            elif silence_start >= first_speech_start and silence_end <= last_speech_end:
                middle_silences.append(silence_seg)
                middle_silence_total += silence_seg["duration"]
        
        # If no leading silence segments detected, but there is time before first speech
        if not leading_silence_segments and first_speech_start > 0:
            leading_silence_total = first_speech_start
        
        # If no trailing silence segments detected, but there is time after last speech
        if not trailing_silence_segments and last_speech_end < total_duration_sec:
            trailing_silence_total = total_duration_sec - last_speech_end
        
        return {
            "leading_silence_sec": round(leading_silence_total, 3),
            "leading_silence_segments": leading_silence_segments,
            "trailing_silence_sec": round(trailing_silence_total, 3),
            "trailing_silence_segments": trailing_silence_segments,
            "middle_silences": middle_silences,
            "middle_silence_total_sec": round(middle_silence_total, 3),
            "middle_silence_count": len(middle_silences)
        }

    def extract_channel_segments(self, channel_idx: int, audio_channel: np.ndarray,
                                 percentile_threshold: int = 30) -> Dict:
        """
        Extract speech and silence segments from a single channel using robust detection.
        """
        try:
            speech_segments = self.__detect_speech_robust(
                audio_channel, self.sr, percentile_threshold=percentile_threshold
            )
            silence_segments = self.__detect_silence_robust(
                audio_channel, self.sr, percentile_threshold=percentile_threshold
            )
            
            duration_sec = len(audio_channel) / self.sr
            speech_duration = sum((seg["end"] - seg["start"]) for seg in speech_segments)
            silence_duration = sum(seg["duration"] for seg in silence_segments)
            
            # Calculate silence types
            silence_types = self.calculate_silence_types(
                speech_segments, 
                silence_segments, 
                duration_sec
            )
            
            return {
                "speech_segments": speech_segments,
                "silence_segments": silence_segments,
                "speech_duration_sec": round(speech_duration, 3),
                "silence_duration_sec": round(silence_duration, 3),
                "total_duration_sec": round(duration_sec, 3),
                "leading_silence_sec": silence_types["leading_silence_sec"],
                "leading_silence_segments": silence_types["leading_silence_segments"],
                "trailing_silence_sec": silence_types["trailing_silence_sec"],
                "trailing_silence_segments": silence_types["trailing_silence_segments"],
                "middle_silences": silence_types["middle_silences"],
                "middle_silence_total_sec": silence_types["middle_silence_total_sec"],
                "middle_silence_count": silence_types["middle_silence_count"]
            }
        except Exception as e:
            print(f"Error processing channel {channel_idx}: {e}")
            return {
                "speech_segments": [],
                "silence_segments": [],
                "speech_duration_sec": 0,
                "silence_duration_sec": 0,
                "total_duration_sec": 0,
                "leading_silence_sec": 0.0,
                "leading_silence_segments": [],
                "trailing_silence_sec": 0.0,
                "trailing_silence_segments": [],
                "middle_silences": [],
                "middle_silence_total_sec": 0.0,
                "middle_silence_count": 0
            }

    def calculate_metrics(self, channel_data: Dict[int, Dict]) -> Dict:
        """
        Calculate balance and naturalness metrics from channel data
        """
        valid_channels = {k: v for k, v in channel_data.items() 
                         if v["speech_duration_sec"] > 0 or v["silence_duration_sec"] > 0}
        
        if not valid_channels:
            return {
                "balance_score": 0.0,
                "balance_assessment": "N/A",
                "naturalness_score": 0.0,
                "channel_ratios": {},
                "total_speech_duration_sec": 0.0,
                "total_silence_duration_sec": 0.0,
                "avg_silence_percentage": 0.0,
                "avg_leading_silence_sec": 0.0,
                "avg_trailing_silence_sec": 0.0,
                "avg_middle_silence_sec": 0.0,
                "total_middle_silence_count": 0,
                "total_leading_silence_segments": 0,
                "total_trailing_silence_segments": 0
            }
        
        num_channels = len(valid_channels)
        
        # Calculate speech durations and ratios
        channel_durations = {ch_idx: data["speech_duration_sec"] 
                           for ch_idx, data in valid_channels.items()}
        total_speech_duration = sum(channel_durations.values())
        
        channel_ratios = {}
        if total_speech_duration > 0:
            for ch_idx, duration in channel_durations.items():
                ratio = duration / total_speech_duration
                channel_ratios[ch_idx] = round(ratio, 4)
        
        # Calculate balance score
        balance_score = 0.0
        if num_channels > 1 and total_speech_duration > 0:
            ideal_time_per_channel = total_speech_duration / num_channels
            deviations = [abs(duration - ideal_time_per_channel) 
                         for duration in channel_durations.values()]
            
            max_possible_deviation = ideal_time_per_channel * (num_channels - 1)
            if max_possible_deviation > 0:
                avg_deviation = sum(deviations) / len(deviations)
                balance_score = 1 - (avg_deviation / max_possible_deviation)
                balance_score = max(0, min(1, balance_score))
        elif num_channels == 1:
            balance_score = 0.0
        
        # Balance assessment
        if balance_score > 0.9:
            balance_assessment = "Perfect balance"
        elif balance_score > 0.7:
            balance_assessment = "Good balance"
        elif balance_score > 0.5:
            balance_assessment = "Moderate balance"
        else:
            balance_assessment = "Poor balance"
        
        # Calculate silence metrics
        total_silence_duration = sum(data["silence_duration_sec"] 
                                    for data in valid_channels.values())
        
        silence_percentages = []
        for data in valid_channels.values():
            total_dur = data["total_duration_sec"]
            if total_dur > 0:
                silence_pct = (data["silence_duration_sec"] / total_dur) * 100
                silence_percentages.append(silence_pct)
        
        avg_silence_percentage = np.mean(silence_percentages) if silence_percentages else 0
        
        # Calculate average silence types across channels
        leading_silences = [data.get("leading_silence_sec", 0) for data in valid_channels.values()]
        trailing_silences = [data.get("trailing_silence_sec", 0) for data in valid_channels.values()]
        middle_silences = [data.get("middle_silence_total_sec", 0) for data in valid_channels.values()]
        middle_silence_counts = [data.get("middle_silence_count", 0) for data in valid_channels.values()]
        
        # Count total segments
        leading_segment_counts = [len(data.get("leading_silence_segments", [])) for data in valid_channels.values()]
        trailing_segment_counts = [len(data.get("trailing_silence_segments", [])) for data in valid_channels.values()]
        
        avg_leading_silence = np.mean(leading_silences) if leading_silences else 0
        avg_trailing_silence = np.mean(trailing_silences) if trailing_silences else 0
        avg_middle_silence = np.mean(middle_silences) if middle_silences else 0
        total_middle_silence_count = sum(middle_silence_counts)
        total_leading_silence_segments = sum(leading_segment_counts)
        total_trailing_silence_segments = sum(trailing_segment_counts)
        
        # Calculate naturalness score
        silence_factor = max(0, 1 - (avg_silence_percentage / 100))
        naturalness_score = balance_score * silence_factor
        
        return {
            "balance_score": round(balance_score, 4),
            "balance_assessment": balance_assessment,
            "naturalness_score": round(naturalness_score, 4),
            "channel_ratios": channel_ratios,
            "total_speech_duration_sec": round(total_speech_duration, 3),
            "total_silence_duration_sec": round(total_silence_duration, 3),
            "avg_silence_percentage": round(avg_silence_percentage, 2),
            "avg_leading_silence_sec": round(avg_leading_silence, 3),
            "avg_trailing_silence_sec": round(avg_trailing_silence, 3),
            "avg_middle_silence_sec": round(avg_middle_silence, 3),
            "total_middle_silence_count": total_middle_silence_count,
            "total_leading_silence_segments": total_leading_silence_segments,
            "total_trailing_silence_segments": total_trailing_silence_segments
        }