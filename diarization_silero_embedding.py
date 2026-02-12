"""
Audio Diarization Module - Silero VAD + Per-Channel Speaker Embeddings

Performs per-channel VAD and speaker clustering, then assigns globally unique
speaker labels across all channels. Returns raw speech segments only.

pip install torch torchaudio librosa speechbrain scikit-learn
"""

import librosa
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
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
        (self.get_speech_timestamps, self.save_audio,
         self.read_audio, self.VADIterator, self.collect_chunks) = self.utils

        # Load speaker embedding model
        self.embeddings_enabled = False
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

    # ------------------------------------------------------------------ #
    #  Silero VAD on a single channel
    # ------------------------------------------------------------------ #
    def _detect_speech_silero(self, audio_channel, sr):
        if sr != 16000:
            audio_16k = librosa.resample(audio_channel, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio_channel

        audio_tensor = torch.from_numpy(audio_16k).float()
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()

        try:
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, self.model,
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

        return [
            {"start": round(seg['start'] / 16000.0, 3),
             "end": round(seg['end'] / 16000.0, 3)}
            for seg in speech_timestamps
        ]

    # ------------------------------------------------------------------ #
    #  Speaker embedding extraction
    # ------------------------------------------------------------------ #
    def _extract_embedding(self, audio_segment, sr):
        if not self.embeddings_enabled:
            return None
        try:
            t = torch.from_numpy(audio_segment).float()
            if t.abs().max() > 1.0:
                t = t / t.abs().max()
            t = t.unsqueeze(0)
            with torch.no_grad():
                emb = self.speaker_model.encode_batch(t)
            return emb.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Warning: embedding extraction failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Estimate number of speakers via silhouette score
    # ------------------------------------------------------------------ #
    @staticmethod
    def _estimate_num_speakers(embeddings, max_speakers=10):
        if len(embeddings) < 2:
            return 1
        best_score, best_n = -1, 1
        for n in range(2, min(len(embeddings), max_speakers) + 1):
            try:
                labels = AgglomerativeClustering(
                    n_clusters=n, linkage='average', metric='cosine'
                ).fit_predict(embeddings)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    if score > best_score:
                        best_score, best_n = score, n
            except Exception:
                continue
        return best_n

    # ------------------------------------------------------------------ #
    #  Per-channel: VAD + embeddings + clustering
    # ------------------------------------------------------------------ #
    def _diarize_channel(self, audio_channel, sr, num_speakers=None):
        # type: (np.ndarray, int, Optional[int]) -> Tuple[List[Dict], int]
        speech_segments = self._detect_speech_silero(audio_channel, sr)
        if not speech_segments:
            return [], 0

        if not self.embeddings_enabled:
            for seg in speech_segments:
                seg["speaker"] = "Speaker_0"
                seg["duration"] = round(seg["end"] - seg["start"], 3)
            return speech_segments, 1

        # Extract embeddings
        embeddings = []
        valid_segments = []
        for seg in speech_segments:
            s = int(seg["start"] * sr)
            e = int(seg["end"] * sr)
            audio_slice = audio_channel[s:e]
            if len(audio_slice) / sr < self.min_segment_duration_for_embedding:
                continue
            emb = self._extract_embedding(audio_slice, sr)
            if emb is not None:
                embeddings.append(emb)
                valid_segments.append(seg)

        if not embeddings:
            for seg in speech_segments:
                seg["speaker"] = "Speaker_0"
                seg["duration"] = round(seg["end"] - seg["start"], 3)
            return speech_segments, 1

        emb_array = np.array(embeddings)

        if num_speakers is None:
            n_spk = self._estimate_num_speakers(emb_array)
        else:
            n_spk = num_speakers

        if n_spk <= 1:
            labels = np.zeros(len(embeddings), dtype=int)
            n_spk = 1
        else:
            try:
                labels = AgglomerativeClustering(
                    n_clusters=n_spk, linkage='average', metric='cosine'
                ).fit_predict(emb_array)
            except Exception as e:
                print(f"Clustering failed: {e}")
                labels = np.zeros(len(embeddings), dtype=int)
                n_spk = 1

        label_map = {}
        for i, lbl in enumerate(labels):
            label_map[id(valid_segments[i])] = int(lbl)

        result_segments = []
        for seg in speech_segments:
            if id(seg) in label_map:
                lbl = label_map[id(seg)]
            else:
                lbl = self._nearest_label_by_time(seg, valid_segments, label_map)
            result_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": f"Speaker_{lbl}",
                "duration": round(seg["end"] - seg["start"], 3)
            })

        return result_segments, n_spk

    @staticmethod
    def _nearest_label_by_time(seg, valid_segments, label_map):
        best_dist = float('inf')
        best_label = 0
        t = seg["start"]
        for vs in valid_segments:
            d = abs(vs["start"] - t)
            if d < best_dist:
                best_dist = d
                best_label = label_map.get(id(vs), 0)
        return best_label

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #
    def run(self, num_speakers=None, num_speakers_per_channel=None):
        # type: (Optional[int], Optional[int]) -> Dict
        """
        Execute per-channel diarization and return globally-labelled speech
        segments sorted by start time.

        Args:
            num_speakers: Alias for num_speakers_per_channel (for CLI compat).
            num_speakers_per_channel: If set, force this many speakers per
                channel. None → auto-detect per channel.

        Returns:
            Dict with speech_segments, num_speakers, speakerRatios,
            speakerChannels, model_parameters, and audio_metadata.
        """
        if num_speakers_per_channel is None and num_speakers is not None:
            num_speakers_per_channel = num_speakers

        total_duration = self.audio.shape[1] / self.sr

        # ---- Step 1: Per-channel diarization ----------------------------
        channel_results = {}  # type: Dict[int, Tuple[List[Dict], int]]
        for ch in range(self.num_channels):
            print(f"  Processing channel {ch}...")
            segs, n_spk = self._diarize_channel(
                self.audio[ch], self.sr, num_speakers_per_channel
            )
            channel_results[ch] = (segs, n_spk)
            print(f"    Found {n_spk} speaker(s), {len(segs)} segments")

        # ---- Step 2: Globally renumber speaker labels --------------------
        global_offset = 0
        all_speech_segments = []
        speaker_durations = {}            # type: Dict[str, float]
        speaker_to_channel = {}           # type: Dict[str, int]

        for ch in range(self.num_channels):
            segs, n_spk = channel_results[ch]

            for seg in segs:
                local_label = int(seg["speaker"].split("_")[1])
                global_label = local_label + global_offset
                global_speaker = f"Speaker_{global_label}"

                dur = round(seg["end"] - seg["start"], 3)

                all_speech_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": global_speaker,
                })

                speaker_durations[global_speaker] = (
                    speaker_durations.get(global_speaker, 0.0) + dur
                )
                speaker_to_channel[global_speaker] = ch

            global_offset += n_spk

        # ---- Step 3: Sort all speech segments by start, then end ---------
        all_speech_segments.sort(key=lambda x: (x["start"], x["end"]))

        # ---- Step 4: Compute global speaker ratios -----------------------
        total_speech = sum(speaker_durations.values())
        speaker_ratios = {}
        for sid, dur in speaker_durations.items():
            speaker_ratios[sid] = round(dur / total_speech, 4) if total_speech > 0 else 0.0

        return {
            "model_parameters": {
                "threshold": self.threshold,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "min_silence_duration_ms": self.min_silence_duration_ms,
                "window_size_samples": 512,
                "speech_pad_ms": self.speech_pad_ms,
                "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
                "clustering_method": "agglomerative",
                "clustering_metric": "cosine",
            },
            "speech_segments": all_speech_segments,
            "num_speakers": global_offset,
            "speakerRatios": speaker_ratios,
            "speakerChannels": speaker_to_channel,
            "audio_metadata": {
                "num_channels": self.num_channels,
                "sample_rate_hz": self.sr,
                "duration_sec": round(total_duration, 3),
            },
        }


# ====================================================================== #
#  CLI usage
# ====================================================================== #
if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
    diarizer = AudioDiarization(path)
    result = diarizer.run()
    print(json.dumps(result, indent=2))