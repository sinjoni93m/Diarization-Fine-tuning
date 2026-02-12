"""
Multi-Speaker Diarization - Silero VAD + Global Speaker Embeddings

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

    print(f"    Speakers detected: {result.get('num_speakers', 0)}")

    ratios = result.get("speakerRatios", {})
    if ratios:
        print(f"    Speaker ratios:")
        for spk, ratio in sorted(ratios.items()):
            print(f"      {spk}: {ratio * 100:.1f}%")

    channels = result.get("speakerChannels", {})
    if channels:
        print(f"    Speaker → channel mapping:")
        for spk, ch in sorted(channels.items()):
            print(f"      {spk}: channel {ch}")

    print(f"    Speech segments: {num_speech}")
    print(f"    Silence segments: {num_silence} (leading={num_leading}, middle={num_middle}, trailing={num_trailing})")
    print(f"    Balance: {metrics.get('balance_score', 0):.3f} ({metrics.get('balance_assessment', 'N/A')})")
    print(f"    Naturalness: {metrics.get('naturalness_score', 0):.3f}")
    print(f"    Avg silence: {metrics.get('avg_silence_percentage', 0):.1f}%")
    print(f"    Total speech: {metrics.get('total_speech_duration_sec', 0):.2f}s")
    print(f"    Total silence: {metrics.get('total_silence_duration_sec', 0):.2f}s")


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
        result = diarizer.run(num_speakers=num_speakers)
        result = convert_numpy_types(result)

        """
        # Overwrite audio_metadata with soundfile info (includes bit_depth)
        result["audio_metadata"] = {
            "num_channels": info.channels,
            "sample_rate_hz": info.samplerate,
            "duration_sec": round(info.duration, 3),
            "bit_depth": info.subtype,
        }
        """

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