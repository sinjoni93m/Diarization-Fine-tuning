"""
Multi-Method Transcript Comparison Module - Using Existing Diarization Files

Takes a parent folder path and processes each subfolder:
1. Finds all audio files and their corresponding diarization JSON files
2. Creates a "subfolder_id_transcripts" folder within each subfolder
3. Generates transcripts using:
   - Whisper Native (no external diarization) — detects language from raw audio using Whisper
   - Custom Diarization: Uses existing WebRTC diarization JSON files — detects language from speech segments using Whisper
   - Custom Diarization: Uses existing Silero diarization JSON files — detects language from speech segments using Whisper

Language detection uses Whisper per-channel:
- Whisper Native: Detects from first 30 seconds of raw audio
- Diarization methods: Detects from actual speech segments (more accurate)

Note: All languages including Vietnamese ('vi') are supported by Whisper.

Usage:
    python3 main_transcript_comparison.py /path/to/parent/folder
    python3 main_transcript_comparison.py /path/to/parent --methods whisper webrtc silero
    python3 main_transcript_comparison.py /path/to/parent --language en
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

# Import transcript generator
from transcript_generator import MultichannelTranscriptGenerator


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


def find_diarization_file(audio_file: Path, method: str) -> Optional[Path]:
    """
    Find the corresponding diarization JSON file for an audio file.
    
    Expected naming patterns:
    - Audio: filename.wav
    - WebRTC: filename_diarization_webrtc_vad.json
    - Silero: filename_diarization_silero_vad.json
    """
    audio_stem = audio_file.stem
    
    if method == 'webrtc':
        pattern = f"{audio_stem}*webrtc*.json"
    elif method == 'silero':
        pattern = f"{audio_stem}*silero*.json"
    else:
        return None
    
    # Search in the same directory as the audio file
    for file in audio_file.parent.glob(pattern):
        return file
    
    return None


def load_diarization_json(diarization_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load and convert diarization JSON to the format expected by transcript generator.
    """
    try:
        with open(diarization_file, 'r', encoding='utf-8') as f:
            diarization_data = json.load(f)
        
        # Convert to the format expected by MultichannelTranscriptGenerator
        # Expected format: {"channel_data": {0: {"speech_segments": [...]}, ...}}
        
        if "speech_segments" in diarization_data:
            # Reorganize by channel
            channel_data = {}
            for segment in diarization_data["speech_segments"]:
                speaker = segment.get("speaker", "channel_0")
                
                # Extract channel index from speaker string (e.g., "channel_0" -> 0)
                if isinstance(speaker, str) and speaker.startswith("channel_"):
                    ch_idx = int(speaker.split("_")[1])
                else:
                    ch_idx = 0
                
                if ch_idx not in channel_data:
                    channel_data[ch_idx] = {"speech_segments": []}
                
                channel_data[ch_idx]["speech_segments"].append({
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            return {
                "channel_data": channel_data,
                "num_channels": diarization_data.get("num_channels", len(channel_data)),
                "sample_rate_hz": diarization_data.get("sample_rate_hz", 44100)
            }
        
        return None
        
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


def generate_transcript_whisper_native(audio_file_path: str, language_hint: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate transcript using Whisper's native diarization.
    Language detection uses SpeechBrain on raw audio (no diarization available yet).
    """
    try:
        print(f"      [WHISPER NATIVE] Generating transcript...")
        
        # No diarization data for Whisper native - detects from raw audio
        generator = MultichannelTranscriptGenerator(
            audio_file_path, 
            language_hint=language_hint,
            diarization_data=None  # Will detect from raw audio
        )
        transcript = generator.transcribe_with_whisper_native()
        
        print(f"        ✓ Generated {transcript.get('num_segments', 0)} segments")
        
        return transcript
        
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {"error": str(e), "transcription_mode": "whisper_native"}


def generate_transcript_with_existing_diarization(audio_file_path: str,
                                                  diarization_file: Path,
                                                  method: str,
                                                  language_hint: str = None,
                                                  verbose: bool = False) -> Dict[str, Any]:
    """
    Generate transcript using existing diarization JSON file.
    Language detection uses SpeechBrain on actual speech segments (more accurate).
    """
    try:
        method_names = {
            'webrtc': 'WebRTC VAD',
            'silero': 'Silero VAD'
        }
        
        print(f"      [{method.upper()}] Loading existing diarization: {diarization_file.name}")
        
        diarization_data = load_diarization_json(diarization_file)
        
        if not diarization_data:
            return {
                "error": f"Failed to load diarization file: {diarization_file}",
                "transcription_mode": "custom_diarization",
                "diarization_method": method
            }
        
        print(f"      [{method.upper()}] Transcribing with {method_names[method]} diarization...")
        
        # Pass diarization data so language detection uses actual speech segments
        generator = MultichannelTranscriptGenerator(
            audio_file_path, 
            language_hint=language_hint,
            diarization_data=diarization_data  # Will detect from speech segments
        )
        transcript = generator.transcribe_with_custom_diarization(diarization_data)
        
        transcript["diarization_method"] = method
        transcript["diarization_method_description"] = method_names[method]
        transcript["diarization_source_file"] = str(diarization_file)
        
        print(f"        ✓ Generated {transcript.get('num_segments', 0)} segments")
        
        return transcript
        
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {
            "error": str(e),
            "transcription_mode": "custom_diarization",
            "diarization_method": method
        }


def process_audio_file(audio_file: Path, 
                      output_dir: Path,
                      methods_to_use: List[str],
                      language_hint: str = None,
                      verbose: bool = False) -> Tuple[int, int]:
    """
    Process a single audio file with all selected transcription methods.
    """
    print(f"    Processing: {audio_file.name}")
    
    try:
        info = sf.info(str(audio_file))
        print(f"      Channels: {info.channels}, Sample Rate: {info.samplerate} Hz, Duration: {info.duration:.2f}s")
    except Exception as e:
        print(f"      ✗ Error reading audio metadata: {e}")
        return 0, len(methods_to_use)
    
    success_count = 0
    error_count = 0
    
    for method in methods_to_use:
        try:
            if method == 'whisper':
                transcript = generate_transcript_whisper_native(
                    str(audio_file),
                    language_hint=language_hint,
                    verbose=verbose
                )
                output_suffix = "whisper_native"
            else:
                # Find existing diarization file
                diarization_file = find_diarization_file(audio_file, method)
                
                if not diarization_file:
                    print(f"      ✗ No {method.upper()} diarization file found for {audio_file.name}")
                    error_count += 1
                    continue
                
                transcript = generate_transcript_with_existing_diarization(
                    str(audio_file),
                    diarization_file=diarization_file,
                    method=method,
                    language_hint=language_hint,
                    verbose=verbose
                )
                output_suffix = f"{method}_diarization"
            
            transcript = convert_numpy_types(transcript)
            
            transcript['audio_file'] = str(audio_file)
            transcript['audio_metadata'] = {
                'num_channels': info.channels,
                'sample_rate_hz': info.samplerate,
                'duration_sec': round(info.duration, 3),
                'bit_depth': info.subtype
            }
            
            output_json_path = output_dir / f"{audio_file.stem}_transcript_{output_suffix}.json"
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            
            if "error" in transcript:
                print(f"        ✗ Error: {transcript['error']}")
                error_count += 1
            else:
                print(f"        ✓ Saved: {output_json_path.name}")
                success_count += 1
                
        except Exception as e:
            print(f"      ✗ Error with {method}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            error_count += 1
    
    return success_count, error_count


def process_subfolder(subfolder: Path,
                     methods_to_use: List[str],
                     language_hint: str = None,
                     verbose: bool = False) -> Tuple[int, int, int]:
    audio_files = find_audio_files(subfolder)
    
    if not audio_files:
        print(f"  No audio files found")
        return 0, 0, 0
    
    print(f"  Found {len(audio_files)} audio file(s)")
    
    transcripts_dir = subfolder / f"{subfolder.name}_transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {transcripts_dir.name}/")
    
    total_success = 0
    total_errors = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n  [{i}/{len(audio_files)}]")
        success, errors = process_audio_file(
            audio_file,
            transcripts_dir,
            methods_to_use,
            language_hint,
            verbose
        )
        total_success += success
        total_errors += errors
    
    return len(audio_files), total_success, total_errors


def main():
    parser = argparse.ArgumentParser(
        description="Generate transcripts using Whisper native and existing diarization JSON files"
    )
    parser.add_argument("parent_path", type=str,
                       help="Path to parent folder containing subfolders with audio files")
    parser.add_argument("--methods", nargs='+',
                       choices=['whisper', 'webrtc', 'silero', 'all'],
                       default=['all'],
                       help="Transcription methods to use (default: all)")
    parser.add_argument("--language", type=str, default=None,
                       help="Language hint for transcription (e.g., 'en', 'es', 'en-US')")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Only whisper, webrtc, silero are active methods
    if 'all' in args.methods:
        methods_to_use = ['whisper', 'webrtc', 'silero']
    else:
        methods_to_use = args.methods
    
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
        print(f"This script processes audio files within subfolders.")
        print(f"Expected structure:")
        print(f"  {parent_path}/")
        print(f"    ├── subfolder1/")
        print(f"    │   ├── audio.wav")
        print(f"    │   ├── audio_diarization_webrtc_vad.json")
        print(f"    │   └── audio_diarization_silero_vad.json")
        print(f"    └── subfolder2/")
        print(f"        ├── audio.wav")
        print(f"        ├── audio_diarization_webrtc_vad.json")
        print(f"        └── audio_diarization_silero_vad.json")
        return
    
    print(f"\n{'='*80}")
    print(f"MULTI-METHOD TRANSCRIPT COMPARISON (Using Existing Diarization Files)")
    print(f"{'='*80}")
    print(f"Parent Folder: {parent_path.absolute()}")
    print(f"Subfolders Found: {len(subfolders)}")
    print(f"Transcription Methods: {', '.join(methods_to_use)}")
    if args.language:
        print(f"Language Hint: {args.language} (overrides detection)")
    else:
        print(f"Language Detection: Whisper per-channel")
        print(f"  - Whisper Native: Detects from raw audio")
        print(f"  - Diarization methods: Detects from speech segments (using existing JSON files)")
    print(f"{'='*80}\n")
    
    total_files = 0
    total_success = 0
    total_errors = 0
    method_stats = {method: {'success': 0, 'error': 0} for method in methods_to_use}
    
    for idx, subfolder in enumerate(subfolders, 1):
        print(f"{'='*80}")
        print(f"[SUBFOLDER {idx}/{len(subfolders)}] {subfolder.name}")
        print(f"{'='*80}")
        
        files_processed, success, errors = process_subfolder(
            subfolder,
            methods_to_use,
            language_hint=args.language,
            verbose=args.verbose
        )
        
        total_files += files_processed
        total_success += success
        total_errors += errors
        
        if files_processed > 0:
            for method in methods_to_use:
                method_success = success // len(methods_to_use)
                method_errors = errors // len(methods_to_use)
                method_stats[method]['success'] += method_success
                method_stats[method]['error'] += method_errors
        
        print(f"\n  Subfolder Summary:")
        print(f"    Files processed: {files_processed}")
        print(f"    Transcripts generated: {success}")
        print(f"    Errors: {errors}")
        print()
    
    print(f"{'='*80}")
    print("TRANSCRIPT COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Total subfolders processed: {len(subfolders)}")
    print(f"Total audio files: {total_files}")
    print(f"Total transcripts generated: {total_success}")
    print(f"Total errors: {total_errors}")
    
    print(f"\nMethod Statistics (approximate):")
    for method in methods_to_use:
        stats = method_stats[method]
        total = stats['success'] + stats['error']
        success_rate = (stats['success'] / total * 100) if total > 0 else 0
        method_label = "Whisper Native" if method == 'whisper' else f"{method.upper()} Diarization"
        print(f"  {method_label:20s}: {stats['success']}/{total} successful ({success_rate:.1f}%)")
    
    if total_success > 0:
        print(f"\n✓ Transcript generation completed!")
        print(f"\nGenerated files per audio file:")
        if 'whisper' in methods_to_use:
            print(f"  - *_transcript_whisper_native.json")
        if 'webrtc' in methods_to_use:
            print(f"  - *_transcript_webrtc_diarization.json")
        if 'silero' in methods_to_use:
            print(f"  - *_transcript_silero_diarization.json")
        
        print(f"\nOutput location:")
        print(f"  Each subfolder contains a '<subfolder_name>_transcripts/' directory")
        print(f"\nLanguage detection:")
        print(f"  - Whisper per-channel language identification")
        print(f"  - Whisper Native: Detects from raw audio (first 30s)")
        print(f"  - WebRTC/Silero: Detects from speech segments in existing diarization files")
        print(f"  - All languages including Vietnamese ('vi') are supported")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()