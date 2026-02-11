"""
Audio Diarization Helper Functions Module - Silero VAD + Speaker Embeddings

Combines Silero VAD for speech detection with speaker embeddings for 
multi-speaker diarization within single audio channels.

=============================================================================
MULTI-SPEAKER DIARIZATION METHODOLOGY (SILERO VAD + EMBEDDINGS)
=============================================================================

This implementation uses a hybrid approach:
1. Silero VAD for accurate speech/silence detection
2. SpeechBrain speaker embeddings (ECAPA-TDNN) for speaker identification
3. Agglomerative clustering to group segments by speaker

ADVANTAGES
----------
✓ Detects multiple speakers within a single channel
✓ Works with mono or multi-channel audio
✓ More accurate than channel-based assumptions
✓ Robust to overlapping speech
✓ Automatic speaker count estimation

INSTALLATION
------------
pip install torch torchaudio speechbrain scikit-learn
"""

import librosa
import numpy as np
import torch
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: speechbrain or sklearn not installed. Multi-speaker detection disabled.")
    print("Install with: pip install speechbrain scikit-learn")


class AudioDiarization:

    def __init__(self, audio_path, 
                 threshold=0.5, 
                 min_speech_duration_ms=250,
                 min_silence_duration_ms=100,
                 speech_pad_ms=30,
                 min_segment_duration_for_embedding=0.5):
        """
        Initialize AudioDiarization with Silero VAD + Speaker Embeddings.
        
        Args:
            audio_path: Path to audio file
            threshold: Speech probability threshold (0.0-1.0, default: 0.5)
            min_speech_duration_ms: Minimum speech duration in milliseconds (default: 250)
            min_silence_duration_ms: Minimum silence gap in milliseconds (default: 100)
            speech_pad_ms: Padding around speech segments in milliseconds (default: 30)
            min_segment_duration_for_embedding: Minimum segment duration in seconds for embedding (default: 0.5)
        """
        self.file_path = audio_path
        self.audio, self.sr = librosa.load(audio_path, sr=None, mono=False)
        
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(1, -1)
        
        self.num_channels = self.audio.shape[0]
        
        # Silero VAD parameters
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_segment_duration_for_embedding = min_segment_duration_for_embedding
        
        # Load Silero VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        # Extract utility functions
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils
        
        # Load speaker embedding model if available
        if EMBEDDINGS_AVAILABLE:
            try:
                print("Loading speaker embedding model (ECAPA-TDNN)...")
                self.speaker_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="tmp_speaker_model",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )
                self.embeddings_enabled = True
                print("Speaker embedding model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load speaker embedding model: {e}")
                self.embeddings_enabled = False
        else:
            self.embeddings_enabled = False

    def __detect_speech_silero(self, audio_channel, sr):
        """
        Detect speech segments using Silero VAD (Deep Learning).
        
        Args:
            audio_channel: Audio signal array (1D numpy array)
            sr: Sample rate
        
        Returns:
            List of speech segment dictionaries with 'start' and 'end' times
        """
        if sr != 16000:
            audio_16k = librosa.resample(audio_channel, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio_channel
        
        audio_tensor = torch.from_numpy(audio_16k).float()
        
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
        
        try:
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.threshold,
                sampling_rate=16000,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                window_size_samples=512,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=False
            )
        except Exception as e:
            print(f"Silero VAD error: {e}")
            return []
        
        speech_segments = []
        for segment in speech_timestamps:
            start_time = segment['start'] / 16000.0
            end_time = segment['end'] / 16000.0
            
            speech_segments.append({
                "start": round(start_time, 3),
                "end": round(end_time, 3)
            })
        
        return speech_segments

    def __extract_speaker_embedding(self, audio_segment: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract speaker embedding from an audio segment.
        
        Args:
            audio_segment: Audio data for the segment
            sr: Sample rate
            
        Returns:
            Speaker embedding vector (numpy array)
        """
        if not self.embeddings_enabled:
            return None
        
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_segment).float()
            
            # Normalize
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
        
        except Exception as e:
            print(f"Warning: Failed to extract embedding: {e}")
            return None

    def __estimate_num_speakers(self, embeddings: np.ndarray, max_speakers: int = 10) -> int:
        """
        Auto-estimate number of speakers using silhouette score.
        
        Args:
            embeddings: Array of speaker embeddings
            max_speakers: Maximum number of speakers to consider
            
        Returns:
            Estimated number of speakers
        """
        if len(embeddings) < 2:
            return 1
        
        best_score = -1
        best_n = 1
        
        max_n = min(len(embeddings), max_speakers)
        
        for n in range(2, max_n + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n,
                    linkage='average',
                    metric='cosine'
                )
                labels = clustering.fit_predict(embeddings)
                
                # Only calculate silhouette if we have multiple clusters
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    
                    if score > best_score:
                        best_score = score
                        best_n = n
            except:
                continue
        
        return best_n

    def __detect_speakers_hybrid(self, audio_channel: np.ndarray, sr: int, 
                                 num_speakers: int = None) -> Tuple[List[Dict], int]:
        """
        Detect speakers by combining VAD + speaker embeddings.
        
        Args:
            audio_channel: Audio signal
            sr: Sample rate
            num_speakers: Expected number of speakers (None for auto-detect)
            
        Returns:
            Tuple of (speaker_segments, estimated_num_speakers)
        """
        # Step 1: Get speech segments from Silero VAD
        speech_segments = self.__detect_speech_silero(audio_channel, sr)
        
        if len(speech_segments) == 0:
            return [], 0
        
        if not self.embeddings_enabled:
            # Fallback: return segments without speaker labels
            return speech_segments, 1
        
        # Step 2: Extract speaker embeddings for each segment
        embeddings = []
        valid_segments = []
        
        for seg in speech_segments:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segment_audio = audio_channel[start_sample:end_sample]
            
            duration = len(segment_audio) / sr
            
            # Skip very short segments
            if duration < self.min_segment_duration_for_embedding:
                continue
            
            # Get embedding
            embedding = self.__extract_speaker_embedding(segment_audio, sr)
            if embedding is not None:
                embeddings.append(embedding)
                valid_segments.append(seg)
        
        if len(embeddings) == 0:
            # No valid embeddings extracted
            return speech_segments, 1
        
        # Step 3: Cluster embeddings to identify speakers
        embeddings_array = np.array(embeddings)
        
        if num_speakers is None:
            # Auto-detect number of speakers
            num_speakers = self.__estimate_num_speakers(embeddings_array)
        
        # If only one speaker detected, return early
        if num_speakers == 1:
            speaker_segments = []
            for seg in valid_segments:
                speaker_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": "Speaker_1",
                    "duration": round(seg["end"] - seg["start"], 3)
                })
            return speaker_segments, 1
        
        # Perform clustering
        try:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='average',
                metric='cosine'
            )
            labels = clustering.fit_predict(embeddings_array)
        except Exception as e:
            print(f"Clustering failed: {e}")
            # Fallback to single speaker
            labels = np.zeros(len(embeddings_array), dtype=int)
            num_speakers = 1
        
        # Step 4: Assign speaker labels to segments
        speaker_segments = []
        for seg, label in zip(valid_segments, labels):
            speaker_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": f"Speaker_{label + 1}",
                "duration": round(seg["end"] - seg["start"], 3)
            })
        
        return sorted(speaker_segments, key=lambda x: x["start"]), num_speakers

    def __detect_silence_from_speech(self, speech_segments: List[Dict], 
                                     total_duration: float) -> List[Dict]:
        """
        Detect silence segments from speech segments (inverse).
        
        Args:
            speech_segments: List of speech segments
            total_duration: Total audio duration
            
        Returns:
            List of silence segments
        """
        silence_segments = []
        
        if not speech_segments:
            if total_duration > 0:
                silence_segments.append({
                    "start": 0.0,
                    "end": round(total_duration, 3),
                    "duration": round(total_duration, 3)
                })
            return silence_segments
        
        speech_segments_sorted = sorted(speech_segments, key=lambda x: x["start"])
        
        # Leading silence
        first_speech_start = speech_segments_sorted[0]["start"]
        if first_speech_start > 0:
            duration = first_speech_start
            min_silence_sec = self.min_silence_duration_ms / 1000.0
            if duration >= min_silence_sec:
                silence_segments.append({
                    "start": 0.0,
                    "end": round(first_speech_start, 3),
                    "duration": round(duration, 3)
                })
        
        # Middle silences
        for i in range(len(speech_segments_sorted) - 1):
            silence_start = speech_segments_sorted[i]["end"]
            silence_end = speech_segments_sorted[i + 1]["start"]
            duration = silence_end - silence_start
            
            min_silence_sec = self.min_silence_duration_ms / 1000.0
            if duration >= min_silence_sec:
                silence_segments.append({
                    "start": round(silence_start, 3),
                    "end": round(silence_end, 3),
                    "duration": round(duration, 3)
                })
        
        # Trailing silence
        last_speech_end = speech_segments_sorted[-1]["end"]
        if last_speech_end < total_duration:
            duration = total_duration - last_speech_end
            min_silence_sec = self.min_silence_duration_ms / 1000.0
            if duration >= min_silence_sec:
                silence_segments.append({
                    "start": round(last_speech_end, 3),
                    "end": round(total_duration, 3),
                    "duration": round(duration, 3)
                })
        
        return silence_segments

    def calculate_silence_types(self, speech_segments: List[Dict], 
                               silence_segments: List[Dict], 
                               total_duration_sec: float) -> Dict:
        """
        Calculate leading, trailing, and middle silences from speech and silence segments.
        """
        if not speech_segments:
            return {
                "leading_silence_sec": round(total_duration_sec, 3),
                "leading_silence_segments": silence_segments,
                "trailing_silence_sec": 0.0,
                "trailing_silence_segments": [],
                "middle_silences": [],
                "middle_silence_total_sec": 0.0,
                "middle_silence_count": 0
            }
        
        speech_segments_sorted = sorted(speech_segments, key=lambda x: x["start"])
        silence_segments_sorted = sorted(silence_segments, key=lambda x: x["start"])
        
        first_speech_start = speech_segments_sorted[0]["start"]
        last_speech_end = speech_segments_sorted[-1]["end"]
        
        leading_silence_segments = []
        trailing_silence_segments = []
        middle_silences = []
        
        leading_silence_total = 0.0
        trailing_silence_total = 0.0
        middle_silence_total = 0.0
        
        for silence_seg in silence_segments_sorted:
            silence_start = silence_seg["start"]
            silence_end = silence_seg["end"]
            
            if silence_end <= first_speech_start:
                leading_silence_segments.append(silence_seg)
                leading_silence_total += silence_seg["duration"]
            elif silence_start >= last_speech_end:
                trailing_silence_segments.append(silence_seg)
                trailing_silence_total += silence_seg["duration"]
            elif silence_start >= first_speech_start and silence_end <= last_speech_end:
                middle_silences.append(silence_seg)
                middle_silence_total += silence_seg["duration"]
        
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
        Extract speech and silence segments with multi-speaker detection.
        
        Args:
            channel_idx: Channel index
            audio_channel: Audio data for the channel
        
        Returns:
            Dictionary containing segment info, speaker data, and multi-speaker metrics
        """
        try:
            # Detect speakers using hybrid approach
            speaker_segments, num_speakers = self.__detect_speakers_hybrid(
                audio_channel, 
                self.sr
            )
            
            # Calculate total duration
            duration_sec = len(audio_channel) / self.sr
            
            # Detect silence from speech segments
            silence_segments = self.__detect_silence_from_speech(speaker_segments, duration_sec)
            
            # Group segments by speaker and calculate ratios
            speakers_data = {}
            total_speech_duration = 0.0
            
            for seg in speaker_segments:
                speaker_id = seg.get("speaker", "Speaker_1")
                if speaker_id not in speakers_data:
                    speakers_data[speaker_id] = {
                        "segments": [],
                        "total_duration": 0.0
                    }
                speakers_data[speaker_id]["segments"].append(seg)
                speakers_data[speaker_id]["total_duration"] += seg["duration"]
                total_speech_duration += seg["duration"]
            
            # Calculate speech ratios for each speaker
            speaker_ratios = {}
            for speaker_id, data in speakers_data.items():
                if total_speech_duration > 0:
                    ratio = data["total_duration"] / total_speech_duration
                    speaker_ratios[speaker_id] = round(ratio, 4)
                else:
                    speaker_ratios[speaker_id] = 0.0
            
            # Calculate silence metrics
            silence_duration = sum(seg["duration"] for seg in silence_segments)
            speech_duration = total_speech_duration
            
            # Categorize silence types
            silence_types = self.calculate_silence_types(
                speaker_segments, 
                silence_segments, 
                duration_sec
            )
            
            # Multi-speaker metadata
            multispeaker_likelihood = num_speakers > 1
            
            return {
                "speech_segments": speaker_segments,
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
                "middle_silence_count": silence_types["middle_silence_count"],
                "speakers": speakers_data,
                "num_speakers": num_speakers,
                "multispeaker_likelihood": multispeaker_likelihood,
                "potential_number_of_speakers_in_channel": num_speakers,
                "speech_ratio_for_all_speakers_across_channel": speaker_ratios
            }
        except Exception as e:
            print(f"Error processing channel {channel_idx}: {e}")
            import traceback
            traceback.print_exc()
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
                "middle_silence_count": 0,
                "speakers": {},
                "num_speakers": 0,
                "multispeaker_likelihood": False,
                "potential_number_of_speakers_in_channel": 0,
                "speech_ratio_for_all_speakers_across_channel": {}
            }

    def calculate_metrics(self, channel_data: Dict[int, Dict]) -> Dict:
        """
        Calculate balance and naturalness metrics from channel data.
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
        
        channel_durations = {ch_idx: data["speech_duration_sec"] 
                           for ch_idx, data in valid_channels.items()}
        total_speech_duration = sum(channel_durations.values())
        
        channel_ratios = {}
        if total_speech_duration > 0:
            for ch_idx, duration in channel_durations.items():
                ratio = duration / total_speech_duration
                channel_ratios[ch_idx] = round(ratio, 4)
        
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
        
        if balance_score > 0.9:
            balance_assessment = "Perfect balance"
        elif balance_score > 0.7:
            balance_assessment = "Good balance"
        elif balance_score > 0.5:
            balance_assessment = "Moderate balance"
        else:
            balance_assessment = "Poor balance"
        
        total_silence_duration = sum(data["silence_duration_sec"] 
                                    for data in valid_channels.values())
        
        silence_percentages = []
        for data in valid_channels.values():
            total_dur = data["total_duration_sec"]
            if total_dur > 0:
                silence_pct = (data["silence_duration_sec"] / total_dur) * 100
                silence_percentages.append(silence_pct)
        
        avg_silence_percentage = np.mean(silence_percentages) if silence_percentages else 0
        
        leading_silences = [data.get("leading_silence_sec", 0) for data in valid_channels.values()]
        trailing_silences = [data.get("trailing_silence_sec", 0) for data in valid_channels.values()]
        middle_silences = [data.get("middle_silence_total_sec", 0) for data in valid_channels.values()]
        middle_silence_counts = [data.get("middle_silence_count", 0) for data in valid_channels.values()]
        
        leading_segment_counts = [len(data.get("leading_silence_segments", [])) for data in valid_channels.values()]
        trailing_segment_counts = [len(data.get("trailing_silence_segments", [])) for data in valid_channels.values()]
        
        avg_leading_silence = np.mean(leading_silences) if leading_silences else 0
        avg_trailing_silence = np.mean(trailing_silences) if trailing_silences else 0
        avg_middle_silence = np.mean(middle_silences) if middle_silences else 0
        total_middle_silence_count = sum(middle_silence_counts)
        total_leading_silence_segments = sum(leading_segment_counts)
        total_trailing_silence_segments = sum(trailing_segment_counts)
        
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