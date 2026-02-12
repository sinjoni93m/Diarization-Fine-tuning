"""
Transcript Generation Module - Using Existing Diarization Files

Takes a parent folder path and processes each subfolder:
1. Finds all audio files and their corresponding diarization JSON files
2. Transcribes full audio with Whisper once, aligns words to diarization segments
3. Saves transcripts directly in the same folder as the audio/diarization files

Usage:
    python3 main_transcript_comparison.py /path/to/parent/folder
    python3 main_transcript_comparison.py /path/to/parent --language es
    python3 main_transcript_comparison.py /path/to/parent --verbose
"""

import os
import argparse
import logging
import json
import numpy as np
import soundfile as sf
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from transcript_generator import TranscriptGenerator


def find_audio_files(folder_path: Path) -> List[Path]:
    audio_files = []
    if folder_path.is_dir():
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.aiff', '.au']:
                audio_files.append(file)
    return sorted(audio_files)


def find_subfolders(parent_path: Path) -> List[Path]:
    subfolders = []
    if parent_path.is_dir():
        for item in parent_path.iterdir():
            if item.is_dir():
                subfolders.append(item)
    return sorted(subfolders)


def find_diarization_file(audio_file: Path) -> Optional[Path]:
    """
    Find the corresponding diarization JSON file for an audio file.
    Looks for: filename*diarization*.json
    """
    audio_stem = audio_file.stem
    pattern = f"{audio_stem}*diarization*.json"

    for file in audio_file.parent.glob(pattern):
        return file

    return None


def load_diarization_json(diarization_file: Path) -> Optional[Dict[str, Any]]:
    """Load diarization JSON. Returns raw data — TranscriptGenerator handles the format."""
    try:
        with open(diarization_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading diarization file {diarization_file}: {e}")
        return None


def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    return obj


def generate_transcript_with_diarization(audio_file_path: str,
                                         diarization_file: Path,
                                         language_hint: str = None,
                                         verbose: bool = False) -> Dict[str, Any]:
    """
    Generate transcript: Whisper full audio transcription + diarization alignment.
    """
    try:
        print(f"      Loading diarization: {diarization_file.name}")

        diarization_data = load_diarization_json(diarization_file)

        if not diarization_data or "speech_segments" not in diarization_data:
            return {
                "error": f"No speech_segments found in: {diarization_file}",
                "transcription_mode": "custom_diarization",
                "diarization_method": "silero_embedding"
            }

        num_segs = len(diarization_data.get("speech_segments", []))
        print(f"      Diarization has {num_segs} speech segments")
        print(f"      Transcribing full audio + aligning to diarization segments...")

        generator = TranscriptGenerator(audio_file_path, language_hint=language_hint)
        transcript = generator.transcribe_with_diarization(diarization_data)

        transcript["diarization_method"] = "silero_embedding"
        transcript["diarization_method_description"] = "Silero+Embedding VAD"
        transcript["diarization_source_file"] = str(diarization_file)

        print(f"        ✓ Generated {transcript.get('num_segments', 0)} transcript segments")

        return transcript

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {
            "error": str(e),
            "transcription_mode": "custom_diarization",
            "diarization_method": "silero_embedding"
        }


def process_audio_file(audio_file: Path,
                       language_hint: str = None,
                       verbose: bool = False) -> Tuple[int, int]:
    """
    Process a single audio file. Saves transcript alongside the audio file.
    """
    print(f"    Processing: {audio_file.name}")

    try:
        info = sf.info(str(audio_file))
        print(f"      Channels: {info.channels}, Sample Rate: {info.samplerate} Hz, Duration: {info.duration:.2f}s")
    except Exception as e:
        print(f"      ✗ Error reading audio metadata: {e}")
        return 0, 1

    diarization_file = find_diarization_file(audio_file)

    if not diarization_file:
        print(f"      ✗ No diarization file found for {audio_file.name}")
        return 0, 1

    try:
        transcript = generate_transcript_with_diarization(
            str(audio_file),
            diarization_file=diarization_file,
            language_hint=language_hint,
            verbose=verbose
        )

        transcript = convert_numpy_types(transcript)

        transcript['audio_file'] = str(audio_file)
        transcript['audio_metadata'] = {
            'num_channels': info.channels,
            'sample_rate_hz': info.samplerate,
            'duration_sec': round(info.duration, 3),
            'bit_depth': info.subtype
        }

        output_json_path = audio_file.parent / f"{audio_file.stem}_transcript_silero_diarization.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        if "error" in transcript:
            print(f"        ✗ Error: {transcript['error']}")
            return 0, 1
        else:
            print(f"        ✓ Saved: {output_json_path.name}")
            return 1, 0

    except Exception as e:
        print(f"      ✗ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 0, 1


def process_subfolder(subfolder: Path,
                      language_hint: str = None,
                      verbose: bool = False) -> Tuple[int, int, int]:
    audio_files = find_audio_files(subfolder)

    if not audio_files:
        print(f"  No audio files found")
        return 0, 0, 0

    print(f"  Found {len(audio_files)} audio file(s)")

    total_success = 0
    total_errors = 0

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n  [{i}/{len(audio_files)}]")
        success, errors = process_audio_file(audio_file, language_hint, verbose)
        total_success += success
        total_errors += errors

    return len(audio_files), total_success, total_errors


def main():
    parser = argparse.ArgumentParser(
        description="Generate transcripts: Whisper full-audio transcription + diarization alignment"
    )
    parser.add_argument("parent_path", type=str,
                        help="Path to parent folder containing subfolders with audio files")
    parser.add_argument("--language", type=str, default=None,
                        help="Language hint for Whisper (e.g., 'en', 'es', 'vi'). "
                             "If not provided, Whisper auto-detects.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging with tracebacks")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    parent_path = Path(args.parent_path)
    if not parent_path.exists():
        print(f"Error: Path not found at {parent_path.absolute()}")
        return

    if not parent_path.is_dir():
        print(f"Error: {parent_path.absolute()} is not a directory")
        return

    subfolders = find_subfolders(parent_path)

    if not subfolders:
        print(f"Error: No subfolders found in {parent_path}")
        print(f"Expected structure:")
        print(f"  {parent_path}/")
        print(f"    ├── subfolder1/")
        print(f"    │   ├── audio.wav")
        print(f"    │   └── audio_*diarization*.json")
        print(f"    └── subfolder2/")
        print(f"        ├── audio.wav")
        print(f"        └── audio_*diarization*.json")
        return

    print(f"\n{'='*80}")
    print(f"TRANSCRIPT GENERATION")
    print(f"{'='*80}")
    print(f"Parent Folder: {parent_path.absolute()}")
    print(f"Subfolders: {len(subfolders)}")
    print(f"Method: Whisper full-audio transcription + diarization segment alignment")
    if args.language:
        print(f"Language: {args.language} (user-provided)")
    else:
        print(f"Language: Whisper auto-detect")
    print(f"{'='*80}\n")

    total_files = 0
    total_success = 0
    total_errors = 0

    for idx, subfolder in enumerate(subfolders, 1):
        print(f"{'='*80}")
        print(f"[SUBFOLDER {idx}/{len(subfolders)}] {subfolder.name}")
        print(f"{'='*80}")

        files_processed, success, errors = process_subfolder(
            subfolder, language_hint=args.language, verbose=args.verbose
        )

        total_files += files_processed
        total_success += success
        total_errors += errors

        print(f"\n  Summary: {success}/{files_processed} succeeded, {errors} errors")
        print()

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Subfolders processed: {len(subfolders)}")
    print(f"Audio files: {total_files}")
    print(f"Transcripts generated: {total_success}")
    print(f"Errors: {total_errors}")

    if total_files > 0:
        print(f"Success rate: {total_success}/{total_files} ({total_success / total_files * 100:.1f}%)")

    if total_success > 0:
        print(f"\nOutput: *_transcript_silero_diarization.json (saved alongside audio files)")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()