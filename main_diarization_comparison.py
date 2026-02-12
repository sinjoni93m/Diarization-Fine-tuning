"""
Multi-Speaker Diarization - Silero VAD + Global Speaker Embeddings

Orchestrates per-channel diarization and computes all derived metrics
(silence, balance, naturalness) from the aggregated speech timeline.

Usage:
    python3 main_diarization.py audio.wav
    python3 main_diarization.py /path/to/folder
    python3 main_diarization.py /path/to/folder --recursive
    python3 main_diarization.py audio.wav --threshold 0.6 --min-speech 300
    python3 main_diarization.py audio.wav --num-speakers 3
    python3 main_diarization.py audio.wav --verbose
"""

import os
import argparse
import json
import numpy as np
import soundfile as sf
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from diarization_silero_embedding import AudioDiarization


# ====================================================================== #
#  Aggregation helpers
# ====================================================================== #

def merge_intervals(segments):
    # type: (List[Dict]) -> List[Tuple[float, float]]
    """Merge speech segments into non-overlapping (start, end) intervals."""
    if not segments:
        return []
    intervals = sorted((seg["start"], seg["end"]) for seg in segments)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def detect_silence(speech_segments, total_duration, min_silence_sec):
    # type: (List[Dict], float, float) -> List[Dict]
    """Find silence gaps from the merged speech timeline."""
    merged = merge_intervals(speech_segments)
    silence = []

    # Leading silence
    first_start = merged[0][0] if merged else total_duration
    if first_start > 0:
        dur = first_start
        if dur >= min_silence_sec:
            silence.append({
                "start": 0.0,
                "end": round(first_start, 3),
                "duration": round(dur, 3),
            })

    # Gaps between merged speech intervals
    for i in range(len(merged) - 1):
        gap_s = merged[i][1]
        gap_e = merged[i + 1][0]
        dur = gap_e - gap_s
        if dur >= min_silence_sec:
            silence.append({
                "start": round(gap_s, 3),
                "end": round(gap_e, 3),
                "duration": round(dur, 3),
            })

    # Trailing silence
    if merged:
        last_end = merged[-1][1]
        if last_end < total_duration:
            dur = total_duration - last_end
            if dur >= min_silence_sec:
                silence.append({
                    "start": round(last_end, 3),
                    "end": round(total_duration, 3),
                    "duration": round(dur, 3),
                })
    elif total_duration > 0:
        silence.append({
            "start": 0.0,
            "end": round(total_duration, 3),
            "duration": round(total_duration, 3),
        })

    return silence


def classify_silence(speech_segments, silence_segments):
    # type: (List[Dict], List[Dict]) -> Dict
    """Categorise silence as leading, trailing, or middle relative to global speech bounds."""
    if not speech_segments:
        return {
            "leading": silence_segments,
            "trailing": [],
            "middle": [],
        }

    first_speech = min(s["start"] for s in speech_segments)
    last_speech = max(s["end"] for s in speech_segments)

    leading, trailing, middle = [], [], []
    for sil in silence_segments:
        if sil["end"] <= first_speech:
            leading.append(sil)
        elif sil["start"] >= last_speech:
            trailing.append(sil)
        else:
            middle.append(sil)

    return {"leading": leading, "trailing": trailing, "middle": middle}


def calculate_balance(speaker_channels, speaker_ratios, num_channels):
    # type: (Dict[str, int], Dict[str, float], int) -> Tuple[float, str]
    """Compute channel balance from speaker durations."""
    if num_channels <= 1:
        return 0.0, "N/A (mono)"

    ch_totals = {}  # type: Dict[int, float]
    for spk, ratio in speaker_ratios.items():
        ch = speaker_channels.get(spk, 0)
        ch_totals[ch] = ch_totals.get(ch, 0.0) + ratio

    total = sum(ch_totals.values())
    if total == 0:
        return 0.0, "Poor balance"

    ideal = total / num_channels
    devs = [abs(ch_totals.get(c, 0.0) - ideal) for c in range(num_channels)]
    max_dev = ideal * (num_channels - 1)
    score = max(0.0, min(1.0, 1 - (sum(devs) / len(devs)) / max_dev)) if max_dev > 0 else 0.0

    if score > 0.9:
        assessment = "Perfect balance"
    elif score > 0.7:
        assessment = "Good balance"
    elif score > 0.5:
        assessment = "Moderate balance"
    else:
        assessment = "Poor balance"

    return round(score, 4), assessment


def build_metrics(diarization_result, silence_segments, silence_types):
    # type: (Dict, List[Dict], Dict) -> Dict
    """Assemble all derived metrics from the diarization result and silence analysis."""
    meta = diarization_result["audio_metadata"]
    total_duration = meta["duration_sec"]
    num_channels = meta["num_channels"]
    speaker_ratios = diarization_result["speakerRatios"]
    speaker_channels = diarization_result["speakerChannels"]

    total_speech = sum(
        round(seg["end"] - seg["start"], 3)
        for seg in diarization_result["speech_segments"]
    )
    total_silence = sum(s["duration"] for s in silence_segments)
    silence_pct = (total_silence / total_duration * 100) if total_duration > 0 else 0.0

    balance_score, balance_assessment = calculate_balance(
        speaker_channels, speaker_ratios, num_channels
    )
    naturalness = balance_score * max(0, 1 - silence_pct / 100)

    return {
        "num_speakers": diarization_result["num_speakers"],
        "speakerRatios": speaker_ratios,
        "speakerChannels": speaker_channels,
        "balance_score": balance_score,
        "balance_assessment": balance_assessment,
        "naturalness_score": round(naturalness, 4),
        "total_speech_duration_sec": round(total_speech, 3),
        "total_silence_duration_sec": round(total_silence, 3),
        "silence_percentage": round(silence_pct, 2),
        "leading_silence_sec": round(sum(s["duration"] for s in silence_types["leading"]), 3),
        "trailing_silence_sec": round(sum(s["duration"] for s in silence_types["trailing"]), 3),
        "middle_silence_sec": round(sum(s["duration"] for s in silence_types["middle"]), 3),
        "middle_silence_count": len(silence_types["middle"]),
    }


# ====================================================================== #
#  Output helpers
# ====================================================================== #

def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    return obj


def print_diarization_summary(result: Dict[str, Any]):
    if "error" in result:
        print(f"    ✗ Error: {result['error']}")
        return

    metrics = result.get("diarization_metrics", {})
    num_speech = len(result.get("speech_segments", []))
    num_silence = len(result.get("all_silence_segments", []))
    num_leading = len(result.get("leading_silence_segments", []))
    num_trailing = len(result.get("trailing_silence_segments", []))
    num_middle = len(result.get("middle_silence_segments", []))

    print(f"    Speakers detected: {metrics.get('num_speakers', 0)}")

    ratios = metrics.get("speakerRatios", {})
    if ratios:
        print(f"    Speaker ratios:")
        for spk, ratio in sorted(ratios.items()):
            print(f"      {spk}: {ratio * 100:.1f}%")

    channels = metrics.get("speakerChannels", {})
    if channels:
        print(f"    Speaker → channel mapping:")
        for spk, ch in sorted(channels.items()):
            print(f"      {spk}: channel {ch}")

    print(f"    Speech segments: {num_speech}")
    print(f"    Silence segments: {num_silence} (leading={num_leading}, middle={num_middle}, trailing={num_trailing})")
    print(f"    Balance: {metrics.get('balance_score', 0):.3f} ({metrics.get('balance_assessment', 'N/A')})")
    print(f"    Naturalness: {metrics.get('naturalness_score', 0):.3f}")
    print(f"    Silence: {metrics.get('silence_percentage', 0):.1f}%")
    print(f"    Total speech: {metrics.get('total_speech_duration_sec', 0):.2f}s")
    print(f"    Total silence: {metrics.get('total_silence_duration_sec', 0):.2f}s")


# ====================================================================== #
#  File / folder processing
# ====================================================================== #

def process_file(audio_path: Path, threshold: float, min_speech_ms: int,
                 min_silence_ms: int, num_speakers: Optional[int],
                 verbose: bool) -> Dict[str, Any]:
    try:
        info = sf.info(str(audio_path))
        print(f"    Channels: {info.channels}, SR: {info.samplerate} Hz, "
              f"Duration: {info.duration:.2f}s, Bit depth: {info.subtype}")

        diarizer = AudioDiarization(
            str(audio_path),
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
        )
        diarization_result = diarizer.run(num_speakers=num_speakers)

        # -- Derive all metrics from aggregated speech segments -----------
        total_duration = diarization_result["audio_metadata"]["duration_sec"]
        min_sil_sec = min_silence_ms / 1000.0
        speech_segments = diarization_result["speech_segments"]

        all_silence = detect_silence(speech_segments, total_duration, min_sil_sec)
        silence_types = classify_silence(speech_segments, all_silence)
        metrics = build_metrics(diarization_result, all_silence, silence_types)

        # -- Assemble final output ----------------------------------------
        result = {
            "model_parameters": diarization_result["model_parameters"],
            "speech_segments": speech_segments,
            "all_silence_segments": all_silence,
            "leading_silence_segments": silence_types["leading"],
            "trailing_silence_segments": silence_types["trailing"],
            "middle_silence_segments": silence_types["middle"],
            "diarization_metrics": metrics,
            "audio_metadata": diarization_result["audio_metadata"],
            "audio_file": str(audio_path),
        }
        result = convert_numpy_types(result)

        # Save JSON
        out_path = audio_path.parent / f"{audio_path.stem}_diarization.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"    ✓ Saved: {out_path.name}")
        print_diarization_summary(result)
        return result

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        print(f"    ✗ Error: {e}")
        return {"error": str(e)}


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".aiff", ".au"}


def find_audio_files(folder: Path) -> List[Path]:
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)


def process_folder(folder: Path, threshold: float, min_speech_ms: int,
                   min_silence_ms: int, num_speakers: Optional[int],
                   verbose: bool) -> Tuple[int, int]:
    files = find_audio_files(folder)
    if not files:
        print(f"  No audio files found in {folder.name}")
        return 0, 0

    print(f"  Found {len(files)} audio file(s)")
    ok, err = 0, 0
    for i, f in enumerate(files, 1):
        print(f"\n  [{i}/{len(files)}] {f.name}")
        res = process_file(f, threshold, min_speech_ms, min_silence_ms, num_speakers, verbose)
        if "error" in res:
            err += 1
        else:
            ok += 1
    return ok, err


def main():
    parser = argparse.ArgumentParser(
        description="Multi-speaker diarization (Silero VAD + ECAPA-TDNN global clustering)"
    )
    parser.add_argument("input_path", help="Audio file or folder")
    parser.add_argument("--threshold", type=float, default=0.5, help="VAD threshold (default: 0.5)")
    parser.add_argument("--min-speech", type=int, default=250, help="Min speech duration ms (default: 250)")
    parser.add_argument("--min-silence", type=int, default=100, help="Min silence duration ms (default: 100)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Fix number of speakers (default: auto-detect)")
    parser.add_argument("--recursive", action="store_true", help="Process subfolders")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    inp = Path(args.input_path)
    if not inp.exists():
        print(f"Error: {inp.absolute()} not found")
        return

    print(f"\n{'=' * 70}")
    print("SILERO VAD + GLOBAL SPEAKER EMBEDDING DIARIZATION")
    print(f"{'=' * 70}")

    if inp.is_file():
        print(f"File: {inp.absolute()}")
        res = process_file(inp, args.threshold, args.min_speech, args.min_silence,
                           args.num_speakers, args.verbose)
        ok = 0 if "error" in res else 1
        err = 1 - ok

    elif inp.is_dir() and args.recursive:
        subs = sorted(d for d in inp.iterdir() if d.is_dir())
        if not subs:
            print("No subfolders found. Try without --recursive.")
            return
        ok, err = 0, 0
        for i, sub in enumerate(subs, 1):
            print(f"\n--- Subfolder [{i}/{len(subs)}]: {sub.name} ---")
            o, e = process_folder(sub, args.threshold, args.min_speech,
                                  args.min_silence, args.num_speakers, args.verbose)
            ok += o
            err += e

    elif inp.is_dir():
        ok, err = process_folder(inp, args.threshold, args.min_speech,
                                 args.min_silence, args.num_speakers, args.verbose)
    else:
        print(f"Error: {inp} is not a valid file or directory")
        return

    print(f"\n{'=' * 70}")
    print(f"Done. Processed: {ok}, Errors: {err}")
    if ok + err > 0:
        print(f"Success rate: {ok / (ok + err) * 100:.0f}%")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()