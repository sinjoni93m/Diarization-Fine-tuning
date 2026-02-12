"""
Transcript Generator - Whisper Full Audio + Diarization Alignment

Simple approach:
1. Transcribe the FULL audio file once with Whisper (word timestamps)
2. Align Whisper's output segments back to diarization speech_segments
3. Each transcript segment inherits the speaker label from diarization

Supports diarization input with flat speech_segments (Speaker_0, Speaker_2, etc.)
"""

import os
import tempfile
import warnings
import numpy as np
import librosa
import soundfile as sf
import whisper
import logging
from typing import Optional, Dict, Any, List, Tuple
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

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


class TranscriptGenerator:
    """
    Transcribe audio by running Whisper once on the full file,
    then aligning word timestamps to diarization speech segments.
    """

    def __init__(self, audio_file_path: str, language_hint: Optional[str] = None):
        self.audio_file_path = audio_file_path
        self.user_language_hint = language_hint
        self.whisper_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_whisper_model(self):
        if self.whisper_model is None:
            print("    Loading Whisper model (large-v3)...")
            self.whisper_model = whisper.load_model("large-v3")

    def _get_language(self) -> str:
        """Get language: use hint if provided, otherwise let Whisper auto-detect."""
        if self.user_language_hint:
            lang = self.user_language_hint.split('-')[0].lower()
            if lang in WHISPER_SUPPORTED_LANGUAGES:
                return lang
            logger.warning(f"Language '{lang}' not supported, falling back to auto-detect")
        return None

    def _transcribe_full_audio(self) -> List[Dict[str, Any]]:
        """
        Transcribe the full audio file with Whisper, returning word-level timestamps.

        Returns:
            List of word dicts: [{"word": str, "start": float, "end": float}, ...]
        """
        self._load_whisper_model()
        lang = self._get_language()

        print(f"    Transcribing full audio with Whisper (language: {lang or 'auto-detect'})...")

        opts = {"verbose": False, "word_timestamps": True}
        if lang:
            opts["language"] = lang

        result = self.whisper_model.transcribe(self.audio_file_path, **opts)

        detected_lang = result.get("language", lang or "en")
        print(f"    Whisper detected language: {detected_lang}")

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"]
                })

        print(f"    Got {len(words)} words from Whisper")
        return words, detected_lang

    @staticmethod
    def _align_words_to_segments(
        words: List[Dict[str, Any]],
        speech_segments: List[Dict[str, Any]],
        tolerance: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Align Whisper words to diarization speech segments.

        Each word is assigned to the diarization segment that overlaps it most.
        Handles overlapping diarization segments (e.g., Speaker_0 and Speaker_2
        at the same time) by picking the segment whose time range best covers
        the word.

        Words outside all segments are dropped (silence hallucinations).

        Args:
            words: [{"word", "start", "end"}, ...] from Whisper
            speech_segments: [{"start", "end", "speaker"}, ...] from diarization
            tolerance: Seconds of tolerance around segment boundaries

        Returns:
            List of transcript segments with text and speaker labels
        """
        if not words or not speech_segments:
            return []

        words_sorted = sorted(words, key=lambda w: w["start"])
        segs_sorted = sorted(speech_segments, key=lambda s: (s["start"], s["end"]))

        # Step 1: Assign each word to its best-matching diarization segment
        # For each word, find all overlapping segments and pick the one with most overlap
        word_assignments = []  # (seg_index, word) pairs

        for w in words_sorted:
            w_start = w["start"]
            w_end = w["end"]
            w_mid = (w_start + w_end) / 2.0

            best_seg_idx = None
            best_overlap = 0.0

            for seg_idx, seg in enumerate(segs_sorted):
                seg_start = seg["start"] - tolerance
                seg_end = seg["end"] + tolerance

                # Quick skip: if segment starts well after word, no more candidates
                if seg_start > w_end + tolerance:
                    break

                # Skip segments that end before word
                if seg_end < w_start - tolerance:
                    continue

                # Calculate overlap between word and segment
                overlap_start = max(w_start, seg_start)
                overlap_end = min(w_end, seg_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_seg_idx = seg_idx
                elif overlap == best_overlap and overlap > 0:
                    # Tie-break: prefer the segment whose center is closer to word midpoint
                    if best_seg_idx is not None:
                        prev_center = (segs_sorted[best_seg_idx]["start"] + segs_sorted[best_seg_idx]["end"]) / 2.0
                        curr_center = (seg["start"] + seg["end"]) / 2.0
                        if abs(curr_center - w_mid) < abs(prev_center - w_mid):
                            best_seg_idx = seg_idx

            if best_seg_idx is not None:
                word_assignments.append((best_seg_idx, w))

        # Step 2: Group words by their assigned segment
        from collections import defaultdict
        seg_words_map = defaultdict(list)
        for seg_idx, w in word_assignments:
            seg_words_map[seg_idx].append(w)

        # Step 3: Build output segments preserving original diarization order
        aligned = []
        for seg_idx, seg in enumerate(segs_sorted):
            seg_word_list = seg_words_map.get(seg_idx, [])
            if not seg_word_list:
                continue

            # Sort words within segment by time
            seg_word_list.sort(key=lambda w: w["start"])
            text = " ".join(w["word"] for w in seg_word_list)
            speaker = seg.get("speaker", "unknown")

            aligned.append({
                "start": seg_word_list[0]["start"],
                "end": seg_word_list[-1]["end"],
                "speaker": speaker,
                "text": text
            })

        return aligned

    def transcribe_with_diarization(self, diarization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transcribe audio using diarization speech segments for speaker labels.

        Steps:
        1. Run Whisper on full audio (one call, full context)
        2. Align word timestamps to diarization segments
        3. Return transcript with speaker labels

        Args:
            diarization_data: Diarization JSON with speech_segments:
                {
                    "speech_segments": [
                        {"start": 5.346, "end": 5.982, "speaker": "Speaker_2"},
                        ...
                    ],
                    "audio_metadata": {...},
                    ...
                }

        Returns:
            Transcript dict matching expected output format
        """
        speech_segments = diarization_data.get("speech_segments", [])
        audio_meta = diarization_data.get("audio_metadata", {})

        if not speech_segments:
            logger.error("No speech_segments in diarization data")
            return {
                "transcription_mode": "custom_diarization",
                "transcript_segments": [],
                "num_segments": 0
            }

        print(f"    Diarization has {len(speech_segments)} speech segments")

        # Step 1: Transcribe full audio once
        words, detected_lang = self._transcribe_full_audio()

        # Step 2: Align words to diarization segments
        print(f"    Aligning {len(words)} words to {len(speech_segments)} diarization segments...")
        aligned = self._align_words_to_segments(words, speech_segments)

        # Add detected language to each segment
        for seg in aligned:
            seg["detected_language"] = detected_lang

        total_words = sum(len(s["text"].split()) for s in aligned)
        dropped = len(words) - total_words
        print(f"    Aligned: {total_words} words into {len(aligned)} segments ({dropped} words dropped)")

        # Collect unique speakers
        speakers = sorted(set(seg["speaker"] for seg in aligned))

        return {
            "transcription_mode": "custom_diarization",
            "primary_language": detected_lang,
            "detected_languages": [detected_lang],
            "language_hint_provided": self.user_language_hint is not None,
            "language_hint": self.user_language_hint,
            "language_source": "user_hint" if self.user_language_hint else "whisper_lid",
            "transcript_segments": aligned,
            "num_segments": len(aligned),
            "speakers": speakers,
            "audio_metadata": audio_meta
        }