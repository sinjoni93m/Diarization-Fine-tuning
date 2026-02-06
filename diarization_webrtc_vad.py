"""
Audio Diarization Helper Functions Module

Contains the AudioDiarization class with helper functions for speech/silence detection
using WebRTC Voice Activity Detection (VAD).

=============================================================================
SPEECH/SILENCE DETECTION METHODOLOGY (WebRTC VAD)
=============================================================================

This implementation uses WebRTC's Voice Activity Detection (VAD) algorithm,
which is significantly more robust than simple energy-based detection and
specifically optimized for real-time communication scenarios.

METHOD: WebRTC VAD (Voice Activity Detection)
----------------------------------------------
WebRTC VAD is a battle-tested algorithm developed by Google for web-based
real-time communication. It uses a combination of:

1. **Energy-based features**: Similar to simple energy detection but more robust
2. **Spectral features**: Analyzes frequency characteristics to distinguish voice
3. **Statistical modeling**: Uses Gaussian Mixture Models (GMMs) for classification
4. **Noise estimation**: Continuously adapts to background noise levels

The algorithm is specifically designed to work well with:
- VoIP/phone audio (compressed, band-limited)
- Real-world noisy environments
- Varying speaker volumes and distances
- Low computational overhead (runs in real-time)

AGGRESSIVENESS MODES
--------------------
WebRTC VAD has 4 aggressiveness levels (0-3):

Mode 0 - QUALITY (Least Aggressive):
   - Most sensitive, detects more speech
   - May include some non-speech sounds
   - Best for: Clean recordings, capturing all speech including quiet parts
   - Use when: You don't want to miss any speech

Mode 1 - LOW_BITRATE:
   - Balanced sensitivity
   - Good general-purpose setting
   - Best for: Typical office/indoor environments
   - Use when: Standard recording conditions

Mode 2 - AGGRESSIVE (Default in this implementation):
   - More strict about what counts as speech
   - Filters out more noise and non-speech sounds
   - Best for: Noisy environments, outdoor recordings
   - Use when: Background noise is present

Mode 3 - VERY_AGGRESSIVE:
   - Most strict, only clear speech is detected
   - May miss quiet or distant speech
   - Best for: Very noisy environments, phone calls
   - Use when: Lots of background noise and you want high precision

AUDIO REQUIREMENTS
------------------
WebRTC VAD has specific technical requirements:

1. **Sample Rate**: Must be 8kHz, 16kHz, 32kHz, or 48kHz
   - This implementation resamples everything to 16kHz (optimal for speech)
   - 16kHz captures 0-8kHz frequency range (covers human speech: 80-8000 Hz)

2. **Audio Format**: 16-bit signed PCM integers
   - Floating-point audio must be converted to int16
   - Range: -32768 to +32767

3. **Frame Duration**: Must be 10ms, 20ms, or 30ms
   - This implementation uses 30ms frames (optimal balance)
   - Longer frames = more context for better accuracy
   - Shorter frames = faster response to speech changes

FRAME DURATION SELECTION
-------------------------
WebRTC supports three frame durations:

10ms frames:
   - Fastest response to speech changes
   - More granular detection
   - Higher computational cost
   - Best for: Real-time applications, capturing quick utterances

20ms frames:
   - Balanced speed and accuracy
   - Good for most applications
   - Moderate computational cost
   - Best for: General-purpose diarization

30ms frames (used in this implementation):
   - Better accuracy (more context per decision)
   - Slightly delayed response
   - Lower computational cost
   - Best for: Offline processing, accuracy over speed

PROCESSING PIPELINE
--------------------
1. **Resampling**: Convert audio to 16kHz if needed
   - Uses librosa's high-quality resampling
   - Preserves speech characteristics

2. **Format Conversion**: Convert float to 16-bit PCM
   - Scales floating-point [-1.0, 1.0] to int16 [-32768, 32767]
   - Clips values outside range to prevent overflow

3. **Frame Extraction**: Split audio into fixed-size frames
   - Each frame must be exactly 480 samples (30ms at 16kHz)
   - No overlap between frames (adjacent frames)

4. **VAD Decision**: Call vad.is_speech() for each frame
   - Returns True if frame contains speech
   - Returns False if frame is silence/noise

5. **Segment Building**: Merge consecutive speech frames
   - Convert boolean frame decisions to time segments
   - Apply minimum duration filters

MINIMUM DURATION FILTERS
-------------------------
After VAD frame-level decisions, we apply duration filters:

1. min_speech_duration (default: 0.3 seconds)
   - Filters out brief noise bursts incorrectly classified as speech
   - Increase to 0.5-1.0 for noisier environments
   - Decrease to 0.1-0.2 to capture short words like "yes", "no"

2. min_silence_duration (default: 0.3 seconds)
   - Filters out brief pauses within continuous speech
   - Increase to 0.5-1.0 to ignore breathing pauses
   - Decrease to 0.1-0.2 to detect all silence gaps

ADVANTAGES OF WebRTC VAD
-------------------------
✓ Much more robust to background noise than energy-based detection
✓ Works well with varying recording conditions
✓ No manual threshold tuning required
✓ Optimized for real-time performance (low CPU usage)
✓ Battle-tested in production VoIP applications
✓ Handles compressed/phone audio well
✓ Adapts to different speaker volumes automatically
✓ Open-source and actively maintained

DISADVANTAGES
-------------
✗ Fixed frame sizes (10/20/30ms) may be too coarse for some applications
✗ Requires specific sample rates (8/16/32/48 kHz)
✗ Less accurate than deep learning models (e.g., Silero VAD)
✗ Aggressiveness modes may need tuning per use case
✗ Cannot distinguish between different speakers (only speech vs silence)
✗ May struggle with whispering or very quiet speech

COMPARISON TO OTHER METHODS
----------------------------
Energy-Based (Simple):
   - WebRTC VAD is significantly more accurate
   - Energy-based fails in noisy environments
   - WebRTC adapts to noise, energy-based uses fixed thresholds

Spectral Features (Moderate):
   - WebRTC VAD is more robust in varied conditions
   - Spectral features require more parameter tuning
   - WebRTC has built-in noise adaptation

Silero VAD (Deep Learning):
   - Silero is more accurate but slower
   - WebRTC is faster and uses less resources
   - WebRTC is better for real-time applications
   - Silero is better for offline, high-accuracy needs

TYPICAL USE CASES
-----------------
✓ VoIP/phone call analysis
✓ Podcast/interview diarization
✓ Meeting transcription systems
✓ Voice command detection
✓ Broadcast audio processing
✓ Real-time speech applications
✓ Mobile/embedded applications (low CPU)

TUNING GUIDELINES
-----------------
If detection is incorrect, try adjusting:

1. **Aggressiveness Level**:
   - More false positives (noise detected as speech) → Increase (2→3)
   - Missing actual speech → Decrease (2→1 or 0)

2. **Minimum Duration Filters**:
   - Too many short speech bursts → Increase min_speech_duration
   - Missing short utterances → Decrease min_speech_duration

3. **Frame Duration**:
   - Need faster response → Use 10ms or 20ms frames
   - Need better accuracy → Use 30ms frames

INSTALLATION
------------
pip install webrtcvad

REFERENCES
----------
- WebRTC VAD Algorithm: Based on GMM statistical models
- Developed by Google for WebRTC project
- Used in Google Duo, Meet, and other communication platforms

=============================================================================
"""

import librosa
import numpy as np
import webrtcvad
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")


class AudioDiarization:

    def __init__(self, audio_path, vad_aggressiveness=2):
        """
        Initialize AudioDiarization with WebRTC VAD.
        
        Args:
            audio_path: Path to audio file
            vad_aggressiveness: WebRTC VAD aggressiveness mode (0-3)
                               0 = Quality (least aggressive, most sensitive)
                               1 = Low Bitrate (balanced)
                               2 = Aggressive (default, good for noisy environments)
                               3 = Very Aggressive (most strict, very noisy environments)
        """
        self.file_path = audio_path
        self.audio, self.sr = librosa.load(audio_path, sr=None, mono=False)
        
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(1, -1)
        
        self.num_channels = self.audio.shape[0]
        self.vad_aggressiveness = vad_aggressiveness

    def __detect_speech_webrtc(self, audio_channel, sr, min_speech_duration=0.3):
        """
        Detect speech segments using WebRTC Voice Activity Detection.
        
        WebRTC VAD is much more robust to noise than simple energy-based detection.
        It uses statistical models and spectral features to distinguish speech
        from noise.
        
        Args:
            audio_channel: Audio signal array
            sr: Sample rate
            min_speech_duration: Minimum duration (seconds) to consider as speech
        
        Returns:
            List of speech segment dictionaries with 'start' and 'end' times
        """
        # Initialize WebRTC VAD with aggressiveness setting
        vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Resample to 16kHz (WebRTC VAD requirement)
        # 16kHz is optimal for speech (captures 0-8kHz frequency range)
        if sr != 16000:
            audio_16k = librosa.resample(audio_channel, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio_channel
        
        # Convert to 16-bit PCM (WebRTC VAD requirement)
        # Scale float [-1.0, 1.0] to int16 [-32768, 32767]
        audio_int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
        
        # WebRTC VAD frame parameters
        # Frame duration must be 10ms, 20ms, or 30ms
        # Using 30ms for best accuracy (more context per frame)
        frame_duration_ms = 30
        frame_length = int(16000 * frame_duration_ms / 1000)  # 480 samples for 30ms
        
        # Process audio in frames and collect VAD decisions
        speech_frames = []
        num_frames = len(audio_int16) // frame_length
        
        for i in range(num_frames):
            # Extract frame
            start_idx = i * frame_length
            end_idx = start_idx + frame_length
            frame = audio_int16[start_idx:end_idx]
            
            # Convert to bytes (required by WebRTC VAD)
            frame_bytes = frame.tobytes()
            
            # Get VAD decision for this frame
            try:
                is_speech = vad.is_speech(frame_bytes, 16000)
                speech_frames.append(is_speech)
            except Exception as e:
                # If frame processing fails, assume silence
                speech_frames.append(False)
        
        # Convert frame-level decisions to time segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            # Convert frame index to time
            t = i * frame_duration_ms / 1000.0
            
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
            t = (num_frames * frame_duration_ms) / 1000.0
            speech_duration = t - speech_start
            if speech_duration >= min_speech_duration:
                speech_segments.append({
                    "start": round(speech_start, 3),
                    "end": round(t, 3)
                })
        
        return speech_segments

    def __detect_silence_webrtc(self, audio_channel, sr, min_silence_duration=0.3):
        """
        Detect silence segments using WebRTC Voice Activity Detection.
        
        This is the inverse of speech detection - frames classified as non-speech
        by WebRTC VAD are considered silence.
        
        Args:
            audio_channel: Audio signal array
            sr: Sample rate
            min_silence_duration: Minimum duration (seconds) to consider as silence
        
        Returns:
            List of silence segment dictionaries with 'start', 'end', and 'duration'
        """
        # Initialize WebRTC VAD
        vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Resample to 16kHz
        if sr != 16000:
            audio_16k = librosa.resample(audio_channel, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio_channel
        
        # Convert to 16-bit PCM
        audio_int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
        
        # WebRTC VAD frame parameters
        frame_duration_ms = 30
        frame_length = int(16000 * frame_duration_ms / 1000)
        
        # Process audio in frames and collect VAD decisions
        silence_frames = []
        num_frames = len(audio_int16) // frame_length
        
        for i in range(num_frames):
            start_idx = i * frame_length
            end_idx = start_idx + frame_length
            frame = audio_int16[start_idx:end_idx]
            frame_bytes = frame.tobytes()
            
            try:
                is_speech = vad.is_speech(frame_bytes, 16000)
                # Silence is the opposite of speech
                silence_frames.append(not is_speech)
            except Exception as e:
                # If frame processing fails, assume silence
                silence_frames.append(True)
        
        # Convert frame-level decisions to time segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            t = i * frame_duration_ms / 1000.0
            
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
            t = (num_frames * frame_duration_ms) / 1000.0
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
        
        # Sort segments by start time
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
        
        # Handle cases where no silence segments were detected but time exists
        if not leading_silence_segments and first_speech_start > 0:
            leading_silence_total = first_speech_start
        
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
        Extract speech and silence segments from a single channel using WebRTC VAD.
        
        Performs complete analysis of one audio channel including:
        - Speech segment detection using WebRTC VAD
        - Silence segment detection (inverse of speech)
        - Duration calculations
        - Silence type categorization
        
        Args:
            channel_idx: Channel index (for error reporting)
            audio_channel: Audio data for the channel (1D numpy array)
        
        Returns:
            Dictionary containing all segment and duration information for the channel
        """
        try:
            # Detect speech and silence using WebRTC VAD
            speech_segments = self.__detect_speech_webrtc(audio_channel, self.sr)
            silence_segments = self.__detect_silence_webrtc(audio_channel, self.sr)
            
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