"""
Audio Diarization Helper Functions Module

Contains the AudioDiarization class with helper functions for speech/silence detection.

=============================================================================
SPEECH/SILENCE DETECTION METHODOLOGY (ENERGY-BASED APPROACH)
=============================================================================

This implementation uses a simple energy-based approach to detect speech and
silence segments. While computationally efficient, it is sensitive to 
background noise and recording conditions.

METHOD: Short-Time Energy Analysis
-----------------------------------
The algorithm measures the energy (power) of the audio signal in short frames
and compares it against a fixed threshold to classify segments as speech or
silence.

ENERGY CALCULATION:
   Energy = sum of squared samples in a frame
   Energy_dB = 10 * log10(Energy)

Frame-by-frame energy is computed and compared to a threshold value measured
in decibels (dB).

FRAME PARAMETERS
----------------
1. Frame Length: 0.025 seconds (25 milliseconds)
   - This is approximately one pitch period for typical human speech
   - Standard in speech processing (used in MFCC extraction, etc.)
   
2. Hop Length: 0.010 seconds (10 milliseconds)
   - 60% overlap between consecutive frames
   - Provides smooth temporal resolution for detecting speech boundaries
   - More overlap = smoother detection, but slower processing

THRESHOLD MECHANISM
-------------------
Fixed Threshold: silence_thresh_db = -40 dB (default)

How it works:
- Frames with energy >= -40 dB are classified as SPEECH
- Frames with energy < -40 dB are classified as SILENCE

LIMITATIONS OF FIXED THRESHOLD:
1. **Sensitive to recording gain**: A quiet recording might have all frames
   below -40 dB even during speech
   
2. **Sensitive to background noise**: Constant background noise (HVAC, traffic,
   computer fans) can push the noise floor above -40 dB, preventing silence
   detection
   
3. **Not adaptive**: The same threshold is used regardless of the specific
   audio characteristics
   
4. **Microphone/environment dependent**: Different microphones and recording
   environments produce vastly different energy distributions

TYPICAL ENERGY LEVELS (for reference):
- Silence/room noise: -60 to -50 dB
- Quiet speech: -45 to -35 dB
- Normal speech: -35 to -20 dB
- Loud speech/shouting: -20 to -5 dB
- Background noise (office): -50 to -40 dB
- Background noise (street): -40 to -30 dB

WHEN TO ADJUST THRESHOLD:
-------------------------
- **Too much speech detected** (silence classified as speech):
  Increase threshold (e.g., -35 dB) to be more strict
  
- **Too much silence detected** (speech classified as silence):
  Decrease threshold (e.g., -45 dB) to be more sensitive
  
- **Noisy environment**: May need thresholds as high as -30 dB
- **Studio recording**: May use thresholds as low as -50 dB

MINIMUM DURATION FILTERS
-------------------------
1. min_speech_duration = 0.3 seconds (default)
   - Filters out brief noise bursts (clicks, pops, keyboard taps)
   - Increase to 0.5-1.0 for noisier environments
   - Decrease to 0.1-0.2 to capture short utterances like "yes", "no"

2. min_silence_duration = 0.3 seconds (default)
   - Filters out brief pauses within speech (breathing, hesitations)
   - Increase to 0.5-1.0 to ignore short pauses in continuous speech
   - Decrease to 0.1-0.2 to detect all silence gaps

SEGMENT POST-PROCESSING
------------------------
After frame-level classification, consecutive speech/silence frames are
merged into segments with start/end times. Segments shorter than the
minimum duration thresholds are filtered out.

SILENCE CATEGORIZATION
----------------------
Detected silence segments are further classified into three types:

1. **Leading Silence**: Silence before the first speech segment
   - Includes recording start time before speaker begins
   
2. **Trailing Silence**: Silence after the last speech segment
   - Includes recording end time after speaker finishes
   
3. **Middle Silence**: Silence between speech segments
   - Natural pauses, turn-taking in conversations, breaths

ADVANTAGES OF THIS APPROACH:
-----------------------------
- Simple and computationally efficient
- Easy to understand and debug
- Works well in clean, controlled recording environments
- No dependencies beyond NumPy

DISADVANTAGES:
--------------
- **High false positive rate in noisy environments**
- Not robust to varying recording conditions
- Requires manual threshold tuning for each environment
- Cannot distinguish between speech and non-speech sounds of similar energy
- Sensitive to recording gain/volume settings

RECOMMENDED USE CASES:
----------------------
- Clean studio recordings
- Controlled recording environments
- Applications where you can manually tune thresholds
- Quick prototyping before implementing more robust methods

NOT RECOMMENDED FOR:
--------------------
- Varying acoustic environments
- Outdoor recordings
- Recordings with background music or noise
- Phone/VoIP audio (compressed, noisy)
- Real-time applications requiring reliability

ALTERNATIVE APPROACHES:
-----------------------
For better noise robustness, consider:
1. Spectral features (spectral centroid, rolloff) - more noise-resistant
2. WebRTC VAD - industry-standard voice activity detection
3. Silero VAD - deep learning based, state-of-the-art accuracy
4. Adaptive thresholding - adjusts to each recording's characteristics

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

    def __detect_speech_segments(self, audio_channel, sr, silence_thresh_db=-40, min_speech_duration=0.3):
        """
        Detect speech (non-silence) segments in an audio channel using energy thresholding.
        
        This method uses short-time energy analysis with a fixed dB threshold.
        Simple but sensitive to background noise and recording conditions.
        
        Args:
            audio_channel: Audio signal array
            sr: Sample rate
            silence_thresh_db: Energy threshold in dB (default: -40)
                              Frames above this are classified as speech
            min_speech_duration: Minimum duration (seconds) to consider as speech
                                Filters out brief noise bursts
        
        Returns:
            List of speech segment dictionaries with 'start' and 'end' times
        """
        # Frame parameters for short-time energy analysis
        frame_length = int(0.025 * sr)  # 25ms frames (standard in speech processing)
        hop_length = int(0.010 * sr)     # 10ms hop (60% overlap for smooth detection)
        
        # Calculate energy for each frame
        energy = np.array([
            np.sum(np.square(audio_channel[i:i+frame_length]))
            for i in range(0, len(audio_channel)-frame_length, hop_length)
        ])
        
        # Convert to decibels (log scale for human perception)
        energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
        
        # Classify frames: energy >= threshold → speech
        speech_frames = energy_db >= silence_thresh_db
        
        # Convert frame-level classification to time segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            t = i * hop_length / sr  # Convert frame index to time
            
            if is_speech and not in_speech:
                # Transition from silence to speech
                speech_start = t
                in_speech = True
            elif not is_speech and in_speech:
                # Transition from speech to silence
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

    def __detect_silence_segments(self, audio_channel, sr, silence_thresh_db=-40, min_silence_duration=0.3):
        """
        Detect silence segments in an audio channel using energy thresholding.
        
        This is the inverse of speech detection - frames below the energy
        threshold are classified as silence.
        
        Args:
            audio_channel: Audio signal array
            sr: Sample rate
            silence_thresh_db: Energy threshold in dB (default: -40)
                              Frames below this are classified as silence
            min_silence_duration: Minimum duration (seconds) to consider as silence
                                 Filters out brief pauses within speech
        
        Returns:
            List of silence segment dictionaries with 'start', 'end', and 'duration'
        """
        # Frame parameters (same as speech detection for consistency)
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        # Calculate energy for each frame
        energy = np.array([
            np.sum(np.square(audio_channel[i:i+frame_length]))
            for i in range(0, len(audio_channel)-frame_length, hop_length)
        ])
        
        # Convert to decibels
        energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
        
        # Classify frames: energy < threshold → silence
        silence_frames = energy_db < silence_thresh_db
        
        # Convert frame-level classification to time segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            t = i * hop_length / sr
            
            if is_silent and not in_silence:
                # Transition from speech to silence
                silence_start = t
                in_silence = True
            elif not is_silent and in_silence:
                # Transition from silence to speech
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
        
        Categorizes detected silence into three types based on their temporal
        relationship to speech segments:
        
        - Leading: Before first speech (recording start delay, intro silence)
        - Trailing: After last speech (recording end delay, outro silence)
        - Middle: Between speech segments (pauses, turn-taking, breaths)
        
        Args:
            speech_segments: List of detected speech segments
            silence_segments: List of detected silence segments
            total_duration_sec: Total audio duration in seconds
        
        Returns:
            Dictionary containing categorized silence information with durations
            and segment lists for each type
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
        
        # Sort segments by start time for chronological processing
        speech_segments_sorted = sorted(speech_segments, key=lambda x: x["start"])
        silence_segments_sorted = sorted(silence_segments, key=lambda x: x["start"])
        
        # Get temporal boundaries of speech activity
        first_speech_start = speech_segments_sorted[0]["start"]
        last_speech_end = speech_segments_sorted[-1]["end"]
        
        # Categorize silences based on temporal position
        leading_silence_segments = []
        trailing_silence_segments = []
        middle_silences = []
        
        leading_silence_total = 0.0
        trailing_silence_total = 0.0
        middle_silence_total = 0.0
        
        for silence_seg in silence_segments_sorted:
            silence_start = silence_seg["start"]
            silence_end = silence_seg["end"]
            
            # Leading silence (before first speech begins)
            if silence_end <= first_speech_start:
                leading_silence_segments.append(silence_seg)
                leading_silence_total += silence_seg["duration"]
            
            # Trailing silence (after last speech ends)
            elif silence_start >= last_speech_end:
                trailing_silence_segments.append(silence_seg)
                trailing_silence_total += silence_seg["duration"]
            
            # Middle silence (between first and last speech)
            elif silence_start >= first_speech_start and silence_end <= last_speech_end:
                middle_silences.append(silence_seg)
                middle_silence_total += silence_seg["duration"]
        
        # Handle cases where no silence segments were detected but time exists
        # (can happen when silence is shorter than min_silence_duration)
        
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

    def extract_channel_segments(self, channel_idx: int, audio_channel: np.ndarray) -> Dict:
        """
        Extract speech and silence segments from a single channel.
        
        Performs complete analysis of one audio channel including:
        - Speech segment detection
        - Silence segment detection
        - Duration calculations
        - Silence type categorization
        
        Args:
            channel_idx: Channel index (for error reporting)
            audio_channel: Audio data for the channel (1D numpy array)
        
        Returns:
            Dictionary containing all segment and duration information for the channel
        """
        try:
            # Detect speech and silence using energy-based method
            speech_segments = self.__detect_speech_segments(audio_channel, self.sr)
            silence_segments = self.__detect_silence_segments(audio_channel, self.sr)
            
            # Calculate durations
            duration_sec = len(audio_channel) / self.sr
            speech_duration = sum((seg["end"] - seg["start"]) for seg in speech_segments)
            silence_duration = sum(seg["duration"] for seg in silence_segments)
            
            # Categorize silence types
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
            # Return empty results on error
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
        Calculate balance and naturalness metrics from channel data.
        
        Analyzes multi-channel audio to compute:
        - Balance score: How evenly speech is distributed across channels
        - Channel ratios: Percentage of total speech time per channel
        - Silence metrics: Average silence percentages and distributions
        - Naturalness score: Combined balance and silence quality metric
        
        Args:
            channel_data: Dictionary mapping channel indices to their segment data
        
        Returns:
            Dictionary containing all calculated metrics and assessments
        """
        # Filter out channels with no audio activity
        valid_channels = {k: v for k, v in channel_data.items() 
                         if v["speech_duration_sec"] > 0 or v["silence_duration_sec"] > 0}
        
        if not valid_channels:
            # Return zero/N/A metrics if no valid channels
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
        
        # Calculate speech durations and ratios per channel
        channel_durations = {ch_idx: data["speech_duration_sec"] 
                           for ch_idx, data in valid_channels.items()}
        total_speech_duration = sum(channel_durations.values())
        
        # Calculate what percentage of total speech each channel contributes
        channel_ratios = {}
        if total_speech_duration > 0:
            for ch_idx, duration in channel_durations.items():
                ratio = duration / total_speech_duration
                channel_ratios[ch_idx] = round(ratio, 4)
        
        # Calculate balance score (0 to 1)
        # Perfect balance = 1.0 (all channels have equal speech time)
        # Poor balance = 0.0 (one channel dominates)
        balance_score = 0.0
        if num_channels > 1 and total_speech_duration > 0:
            # Ideal: each channel gets equal time
            ideal_time_per_channel = total_speech_duration / num_channels
            
            # Measure deviation from ideal
            deviations = [abs(duration - ideal_time_per_channel) 
                         for duration in channel_durations.values()]
            
            # Normalize by maximum possible deviation
            max_possible_deviation = ideal_time_per_channel * (num_channels - 1)
            if max_possible_deviation > 0:
                avg_deviation = sum(deviations) / len(deviations)
                balance_score = 1 - (avg_deviation / max_possible_deviation)
                balance_score = max(0, min(1, balance_score))
        elif num_channels == 1:
            # Single channel has no balance concept
            balance_score = 0.0
        
        # Human-readable balance assessment
        if balance_score > 0.9:
            balance_assessment = "Perfect balance"
        elif balance_score > 0.7:
            balance_assessment = "Good balance"
        elif balance_score > 0.5:
            balance_assessment = "Moderate balance"
        else:
            balance_assessment = "Poor balance"
        
        # Calculate silence statistics across all channels
        total_silence_duration = sum(data["silence_duration_sec"] 
                                    for data in valid_channels.values())
        
        # Calculate silence as percentage of total duration for each channel
        silence_percentages = []
        for data in valid_channels.values():
            total_dur = data["total_duration_sec"]
            if total_dur > 0:
                silence_pct = (data["silence_duration_sec"] / total_dur) * 100
                silence_percentages.append(silence_pct)
        
        avg_silence_percentage = np.mean(silence_percentages) if silence_percentages else 0
        
        # Calculate average silence types across all channels
        leading_silences = [data.get("leading_silence_sec", 0) for data in valid_channels.values()]
        trailing_silences = [data.get("trailing_silence_sec", 0) for data in valid_channels.values()]
        middle_silences = [data.get("middle_silence_total_sec", 0) for data in valid_channels.values()]
        middle_silence_counts = [data.get("middle_silence_count", 0) for data in valid_channels.values()]
        
        # Count total number of silence segments by type
        leading_segment_counts = [len(data.get("leading_silence_segments", [])) for data in valid_channels.values()]
        trailing_segment_counts = [len(data.get("trailing_silence_segments", [])) for data in valid_channels.values()]
        
        avg_leading_silence = np.mean(leading_silences) if leading_silences else 0
        avg_trailing_silence = np.mean(trailing_silences) if trailing_silences else 0
        avg_middle_silence = np.mean(middle_silences) if middle_silences else 0
        total_middle_silence_count = sum(middle_silence_counts)
        total_leading_silence_segments = sum(leading_segment_counts)
        total_trailing_silence_segments = sum(trailing_segment_counts)
        
        # Calculate naturalness score
        # Combines balance with silence characteristics
        # Lower silence percentage = more natural (continuous speech)
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