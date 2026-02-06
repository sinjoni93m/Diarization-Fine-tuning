"""
Audio Diarization Helper Functions Module

Contains the AudioDiarization class with helper functions for speech/silence detection.
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
        Detect speech (non-silence) segments in an audio channel.
        """
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        energy = np.array([
            np.sum(np.square(audio_channel[i:i+frame_length]))
            for i in range(0, len(audio_channel)-frame_length, hop_length)
        ])
        energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
        
        speech_frames = energy_db >= silence_thresh_db
        
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            t = i * hop_length / sr
            if is_speech and not in_speech:
                speech_start = t
                in_speech = True
            elif not is_speech and in_speech:
                speech_duration = t - speech_start
                if speech_duration >= min_speech_duration:
                    speech_segments.append({
                        "start": round(speech_start, 3),
                        "end": round(t, 3)
                    })
                in_speech = False
        
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
        Detect silence segments in an audio channel
        """
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        energy = np.array([
            np.sum(np.square(audio_channel[i:i+frame_length]))
            for i in range(0, len(audio_channel)-frame_length, hop_length)
        ])
        energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
        
        silence_frames = energy_db < silence_thresh_db
        
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            t = i * hop_length / sr
            if is_silent and not in_silence:
                silence_start = t
                in_silence = True
            elif not is_silent and in_silence:
                silence_duration = t - silence_start
                if silence_duration >= min_silence_duration:
                    silence_segments.append({
                        "start": round(silence_start, 3),
                        "end": round(t, 3),
                        "duration": round(silence_duration, 3)
                    })
                in_silence = False
        
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

    def extract_channel_segments(self, channel_idx: int, audio_channel: np.ndarray) -> Dict:
        """Extract speech and silence segments from a single channel"""
        try:
            speech_segments = self.__detect_speech_segments(audio_channel, self.sr)
            silence_segments = self.__detect_silence_segments(audio_channel, self.sr)
            
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