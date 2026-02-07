"""
Multi-Method Transcript Comparison Module

Takes a parent folder path and processes each subfolder:
1. Finds all audio files in each subfolder
2. Creates a "subfolder_id_transcripts" folder within each subfolder
3. Generates transcripts using:
   - Whisper Native (no external diarization)
   - Custom Diarization: Energy-Based
   - Custom Diarization: Spectral Features
   - Custom Diarization: WebRTC VAD
   - Custom Diarization: Silero VAD

Usage:
    # Process all subfolders recursively
    python3 main_transcript_comparison.py /path/to/parent/folder
    
    # Process with specific methods only
    python3 main_transcript_comparison.py /path/to/parent --methods whisper energy silero
    
    # With language hint
    python3 main_transcript_comparison.py /path/to/parent --language en
    
    # Verbose output
    python3 main_transcript_comparison.py /path/to/parent --verbose
"""

import os
import argparse
import logging
import json
import numpy as np
import soundfile as sf
from typing import Dict, Any, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import diarization methods
from diarization_old import AudioDiarization as AudioDiarizationEnergy
from diarization_spectral_adaptive import AudioDiarization as AudioDiarizationSpectral
from diarization_webrtc_vad import AudioDiarization as AudioDiarizationWebRTC
from diarization_silero import AudioDiarization as AudioDiarizationSilero

# Import transcript generator
from transcript_generator import MultichannelTranscriptGenerator


def find_audio_files(folder_path: Path) -> List[Path]:
    """
    Find all audio files in the given folder (non-recursive).
    """
    audio_files = []
    if folder_path.is_dir():
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.aiff', '.au']:
                audio_files.append(file)
    return sorted(audio_files)


def find_subfolders(parent_path: Path) -> List[Path]:
    """
    Find all immediate subfolders in the given parent folder.
    """
    subfolders = []
    if parent_path.is_dir():
        for item in parent_path.iterdir():
            if item.is_dir():
                subfolders.append(item)
    return sorted(subfolders)


def convert_numpy_types(obj):
    """
    Convert numpy types to JSON-serializable types.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    return obj


def generate_diarization_for_transcript(audio_file_path: str, method: str, verbose: bool) -> Dict[str, Any]:
    """
    Generate diarization for a specific method.
    Returns diarization data needed for transcription.
    """
    try:
        if method == 'energy':
            diarizer = AudioDiarizationEnergy(audio_file_path)
        elif method == 'spectral':
            diarizer = AudioDiarizationSpectral(audio_file_path)
        elif method == 'webrtc':
            diarizer = AudioDiarizationWebRTC(audio_file_path, vad_aggressiveness=2)
        elif method == 'silero':
            diarizer = AudioDiarizationSilero(audio_file_path, threshold=0.5)
        else:
            return {"error": f"Unknown method: {method}"}
        
        # Extract channel data for transcription
        channel_data = {}
        for ch_idx in range(diarizer.num_channels):
            channel_audio = diarizer.audio[ch_idx]
            channel_info = diarizer.extract_channel_segments(ch_idx, channel_audio)
            channel_data[ch_idx] = channel_info
        
        return {
            "channel_data": channel_data,
            "num_channels": diarizer.num_channels,
            "sample_rate_hz": diarizer.sr
        }
        
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {"error": str(e)}


def generate_transcript_whisper_native(audio_file_path: str, language_hint: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate transcript using Whisper's native diarization (no external VAD).
    """
    try:
        print(f"      [WHISPER NATIVE] Generating transcript...")
        
        generator = MultichannelTranscriptGenerator(audio_file_path, language_hint=language_hint)
        transcript = generator.transcribe_with_whisper_native()
        
        print(f"        ✓ Generated {transcript.get('num_segments', 0)} segments")
        print(f"        Language: {transcript.get('primary_language', 'unknown')}")
        
        return transcript
        
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return {"error": str(e), "transcription_mode": "whisper_native"}


def generate_transcript_with_custom_diarization(audio_file_path: str, 
                                               method: str,
                                               language_hint: str = None,
                                               verbose: bool = False) -> Dict[str, Any]:
    """
    Generate transcript using custom diarization method.
    """
    try:
        method_names = {
            'energy': 'Energy-Based',
            'spectral': 'Spectral Features',
            'webrtc': 'WebRTC VAD',
            'silero': 'Silero VAD'
        }
        
        print(f"      [{method.upper()}] Generating diarization...")
        
        # Generate diarization
        diarization_data = generate_diarization_for_transcript(audio_file_path, method, verbose)
        
        if "error" in diarization_data:
            return {
                "error": diarization_data["error"],
                "transcription_mode": "custom_diarization",
                "diarization_method": method
            }
        
        print(f"      [{method.upper()}] Transcribing with {method_names[method]} diarization...")
        
        # Generate transcript
        generator = MultichannelTranscriptGenerator(audio_file_path, language_hint=language_hint)
        transcript = generator.transcribe_with_custom_diarization(diarization_data)
        
        # Add diarization method info
        transcript["diarization_method"] = method
        transcript["diarization_method_description"] = method_names[method]
        
        print(f"        ✓ Generated {transcript.get('num_segments', 0)} segments")
        print(f"        Language: {transcript.get('primary_language', 'unknown')}")
        
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
    
    Returns:
        Tuple of (success_count, error_count)
    """
    print(f"    Processing: {audio_file.name}")
    
    # Get audio metadata
    try:
        info = sf.info(str(audio_file))
        print(f"      Channels: {info.channels}, Sample Rate: {info.samplerate} Hz, Duration: {info.duration:.2f}s")
    except Exception as e:
        print(f"      ✗ Error reading audio metadata: {e}")
        return 0, len(methods_to_use)
    
    success_count = 0
    error_count = 0
    
    # Process each method
    for method in methods_to_use:
        try:
            # Generate transcript based on method
            if method == 'whisper':
                transcript = generate_transcript_whisper_native(
                    str(audio_file),
                    language_hint=language_hint,
                    verbose=verbose
                )
                output_suffix = "whisper_native"
            else:
                transcript = generate_transcript_with_custom_diarization(
                    str(audio_file),
                    method=method,
                    language_hint=language_hint,
                    verbose=verbose
                )
                output_suffix = f"{method}_diarization"
            
            # Convert numpy types
            transcript = convert_numpy_types(transcript)
            
            # Add audio file metadata
            transcript['audio_file'] = str(audio_file)
            transcript['audio_metadata'] = {
                'num_channels': info.channels,
                'sample_rate_hz': info.samplerate,
                'duration_sec': round(info.duration, 3),
                'bit_depth': info.subtype
            }
            
            # Save transcript JSON
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
    """
    Process all audio files in a single subfolder.
    
    Returns:
        Tuple of (files_processed, success_count, error_count)
    """
    # Find audio files in subfolder
    audio_files = find_audio_files(subfolder)
    
    if not audio_files:
        print(f"  No audio files found")
        return 0, 0, 0
    
    print(f"  Found {len(audio_files)} audio file(s)")
    
    # Create transcripts output directory
    transcripts_dir = subfolder / f"{subfolder.name}_transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {transcripts_dir.name}/")
    
    total_success = 0
    total_errors = 0
    
    # Process each audio file
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
    """
    Main function to perform multi-method transcript comparison.
    """
    parser = argparse.ArgumentParser(
        description="Generate transcripts using multiple methods (Whisper native + custom diarization)"
    )
    parser.add_argument("parent_path", type=str,
                       help="Path to parent folder containing subfolders with audio files")
    parser.add_argument("--methods", nargs='+',
                       choices=['whisper', 'energy', 'spectral', 'webrtc', 'silero', 'all'],
                       default=['all'],
                       help="Transcription methods to use (default: all)")
    parser.add_argument("--language", type=str, default=None,
                       help="Language hint for transcription (e.g., 'en', 'es', 'en-US')")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Determine which methods to use
    if 'all' in args.methods:
        methods_to_use = ['whisper', 'energy', 'spectral', 'webrtc', 'silero']
    else:
        methods_to_use = args.methods
    
    # Validate parent path
    parent_path = Path(args.parent_path)
    if not parent_path.exists():
        print(f"Error: Path not found at {parent_path.absolute()}")
        return
    
    if not parent_path.is_dir():
        print(f"Error: {parent_path.absolute()} is not a directory")
        return
    
    # Find subfolders
    subfolders = find_subfolders(parent_path)
    
    if not subfolders:
        print(f"Error: No subfolders found in {parent_path}")
        print(f"This script processes audio files within subfolders.")
        print(f"Expected structure:")
        print(f"  {parent_path}/")
        print(f"    ├── subfolder1/")
        print(f"    │   └── audio.wav")
        print(f"    └── subfolder2/")
        print(f"        └── audio.wav")
        return
    
    # Print header
    print(f"\n{'='*80}")
    print(f"MULTI-METHOD TRANSCRIPT COMPARISON")
    print(f"{'='*80}")
    print(f"Parent Folder: {parent_path.absolute()}")
    print(f"Subfolders Found: {len(subfolders)}")
    print(f"Transcription Methods: {', '.join(methods_to_use)}")
    if args.language:
        print(f"Language Hint: {args.language}")
    print(f"{'='*80}\n")
    
    # Process each subfolder
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
        
        # Update per-method stats (approximate)
        if files_processed > 0:
            for method in methods_to_use:
                # Approximate: distribute successes/errors evenly across methods
                method_success = success // len(methods_to_use)
                method_errors = errors // len(methods_to_use)
                method_stats[method]['success'] += method_success
                method_stats[method]['error'] += method_errors
        
        print(f"\n  Subfolder Summary:")
        print(f"    Files processed: {files_processed}")
        print(f"    Transcripts generated: {success}")
        print(f"    Errors: {errors}")
        print()
    
    # Print final summary
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
        if 'energy' in methods_to_use:
            print(f"  - *_transcript_energy_diarization.json")
        if 'spectral' in methods_to_use:
            print(f"  - *_transcript_spectral_diarization.json")
        if 'webrtc' in methods_to_use:
            print(f"  - *_transcript_webrtc_diarization.json")
        if 'silero' in methods_to_use:
            print(f"  - *_transcript_silero_diarization.json")
        
        print(f"\nOutput location:")
        print(f"  Each subfolder contains a '<subfolder_name>_transcripts/' directory")
        print(f"\nEach transcript file contains:")
        print(f"  ├── transcription_mode (whisper_native or custom_diarization)")
        print(f"  ├── diarization_method (for custom mode)")
        print(f"  ├── primary_language")
        print(f"  ├── transcript_segments (with timestamps, speaker, text)")
        print(f"  └── audio_metadata")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()