"""
Silero VAD + Speaker Embeddings Multi-Speaker Diarization

Performs multi-speaker diarization on audio files using Silero VAD for speech
detection combined with ECAPA-TDNN speaker embeddings for speaker identification.

Usage: 
    # Single audio file:
    python3 main_silero_multispeaker.py audio.wav
    
    # Folder (non-recursive):
    python3 main_silero_multispeaker.py /path/to/audio/folder
    
    # Folder (recursive - process all subfolders):
    python3 main_silero_multispeaker.py /path/to/parent/folder --recursive
    
    # Adjust parameters:
    python3 main_silero_multispeaker.py audio.wav --threshold 0.6 --min-speech 300
    
    # Verbose output:
    python3 main_silero_multispeaker.py audio.wav --verbose
"""

import os
import argparse
import logging
import numpy as np
import soundfile as sf
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import the multi-speaker diarization module
from diarization_silero_mulri_speaker import AudioDiarization


def convert_numpy_types(obj):
    """
    Convert numpy types to JSON-serializable types
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    return obj


def print_speaker_summary(channel_data: Dict):
    """
    Print a summary of detected speakers per channel
    """
    for ch_idx, ch_data in channel_data.items():
        num_speakers = ch_data.get("num_speakers", 0)
        multispeaker = ch_data.get("multispeaker_likelihood", False)
        speaker_ratios = ch_data.get("speech_ratio_for_all_speakers_across_channel", {})
        
        print(f"      Channel {ch_idx}:")
        print(f"        Speakers detected: {num_speakers}")
        print(f"        Multi-speaker: {'Yes' if multispeaker else 'No'}")
        
        if speaker_ratios:
            print(f"        Speech distribution:")
            for speaker, ratio in sorted(speaker_ratios.items()):
                percentage = ratio * 100
                print(f"          {speaker}: {percentage:.1f}%")


def print_diarization_summary(result: Dict[str, Any]):
    """
    Print a summary of diarization results
    """
    if "error" in result:
        print(f"    ✗ Error: {result['error']}")
        return
    
    metrics = result.get('diarization_metrics', {})
    num_speech = len(result.get('speech_segments', []))
    num_silence = len(result.get('all_silence_segments', []))
    num_leading = len(result.get('leading_silence_segments', []))
    num_trailing = len(result.get('trailing_silence_segments', []))
    num_middle = len(result.get('middle_silence_segments', []))
    
    print(f"    Speech segments: {num_speech}")
    print(f"    Total silence segments: {num_silence}")
    print(f"      - Leading: {num_leading} ({metrics.get('avg_leading_silence_sec', 0):.2f}s avg)")
    print(f"      - Middle: {num_middle} ({metrics.get('avg_middle_silence_sec', 0):.2f}s avg)")
    print(f"      - Trailing: {num_trailing} ({metrics.get('avg_trailing_silence_sec', 0):.2f}s avg)")
    print(f"    Balance: {metrics.get('balance_score', 0):.3f} ({metrics.get('balance_assessment', 'N/A')})")
    print(f"    Naturalness: {metrics.get('naturalness_score', 0):.3f}")
    print(f"    Avg silence: {metrics.get('avg_silence_percentage', 0):.1f}%")


def generate_diarization_silero_multispeaker(audio_file_path: str, 
                                             threshold: float = 0.5,
                                             min_speech_duration_ms: int = 250,
                                             min_silence_duration_ms: int = 100,
                                             verbose: bool = False) -> Dict[str, Any]:
    """
    Generate diarization using Silero VAD + Speaker Embeddings.
    
    Args:
        audio_file_path: Path to audio file
        threshold: VAD threshold (0.0-1.0)
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence gap duration
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing all diarization results
    """
    try:
        if verbose:
            print("    Method: Silero VAD + Speaker Embeddings")
            print("    - Deep learning VAD + ECAPA-TDNN embeddings")
            print("    - Multi-speaker detection within channels")
        
        diarizer = AudioDiarization(
            audio_file_path,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )
        
        all_speech_segments = []
        all_silence_segments = []
        all_leading_silence_segments = []
        all_trailing_silence_segments = []
        all_middle_silence_segments = []
        channel_data = {}
        
        # Extract segments from each channel
        for ch_idx in range(diarizer.num_channels):
            channel_audio = diarizer.audio[ch_idx]
            channel_info = diarizer.extract_channel_segments(ch_idx, channel_audio)
            channel_data[ch_idx] = channel_info
            
            # Add speech segments with speaker labels
            for segment in channel_info["speech_segments"]:
                all_speech_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment.get("speaker", f"channel_{ch_idx}_speaker_unknown"),
                    "channel": ch_idx
                })
            
            # Add silence segments
            for segment in channel_info["silence_segments"]:
                all_silence_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "channel": ch_idx
                })
            
            # Add categorized silence segments
            for segment in channel_info.get("leading_silence_segments", []):
                all_leading_silence_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "channel": ch_idx
                })
            
            for segment in channel_info.get("trailing_silence_segments", []):
                all_trailing_silence_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "channel": ch_idx
                })
            
            for segment in channel_info.get("middle_silences", []):
                all_middle_silence_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "channel": ch_idx
                })
        
        # Sort segments
        all_speech_segments.sort(key=lambda x: x["start"])
        all_silence_segments.sort(key=lambda x: x["start"])
        all_leading_silence_segments.sort(key=lambda x: x["start"])
        all_trailing_silence_segments.sort(key=lambda x: x["start"])
        all_middle_silence_segments.sort(key=lambda x: x["start"])
        
        # Calculate metrics
        metrics = diarizer.calculate_metrics(channel_data)
        
        # Add multi-speaker metadata per channel
        channel_metadata = {}
        for ch_idx, ch_data in channel_data.items():
            channel_metadata[f"channel_{ch_idx}"] = {
                "multispeaker_likelihood": ch_data.get("multispeaker_likelihood", False),
                "potential_number_of_speakers_in_channel": ch_data.get("potential_number_of_speakers_in_channel", 0),
                "speech_ratio_for_all_speakers_across_channel": ch_data.get("speech_ratio_for_all_speakers_across_channel", {})
            }
        
        return {
            "method": "silero_vad_multispeaker",
            "method_description": "Silero VAD + Speaker Embeddings (ECAPA-TDNN) with agglomerative clustering",
            "parameters": {
                "threshold": threshold,
                "min_speech_duration_ms": min_speech_duration_ms,
                "min_silence_duration_ms": min_silence_duration_ms,
                "window_size_samples": 512,
                "speech_pad_ms": 30,
                "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
                "clustering_method": "agglomerative",
                "clustering_metric": "cosine"
            },
            "speech_segments": all_speech_segments,
            "all_silence_segments": all_silence_segments,
            "leading_silence_segments": all_leading_silence_segments,
            "trailing_silence_segments": all_trailing_silence_segments,
            "middle_silence_segments": all_middle_silence_segments,
            "diarization_metrics": metrics,
            "num_channels": diarizer.num_channels,
            "sample_rate_hz": diarizer.sr,
            "multispeaker_metadata": channel_metadata,
            "channel_data": channel_data  # Include full channel data for summary
        }
        
    except Exception as e:
        import traceback
        if verbose:
            traceback.print_exc()
        return {"error": str(e), "method": "silero_vad_multispeaker"}


def find_audio_files(folder_path: Path) -> List[Path]:
    """
    Find all audio files in the given folder (non-recursive)
    """
    audio_files = []
    if folder_path.is_dir():
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.aiff', '.au']:
                audio_files.append(file)
    return sorted(audio_files)


def find_subfolders(parent_path: Path) -> List[Path]:
    """
    Find all immediate subfolders in the given parent folder
    """
    subfolders = []
    if parent_path.is_dir():
        for item in parent_path.iterdir():
            if item.is_dir():
                subfolders.append(item)
    return sorted(subfolders)


def process_folder(folder_path: Path, threshold: float, min_speech_ms: int, 
                   min_silence_ms: int, verbose: bool) -> Tuple[int, int]:
    """
    Process all audio files in a single folder.
    
    Returns:
        Tuple of (processed_count, error_count)
    """
    audio_files = find_audio_files(folder_path)
    
    if not audio_files:
        print(f"  No audio files found in {folder_path.name}")
        return 0, 0
    
    print(f"  Found {len(audio_files)} audio file(s)")
    
    processed_count = 0
    error_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n  [{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            # Get audio metadata
            info = sf.info(str(audio_file))
            print(f"    Channels: {info.channels}, Sample Rate: {info.samplerate} Hz, Duration: {info.duration:.2f}s")
            
            # Generate diarization
            result = generate_diarization_silero_multispeaker(
                str(audio_file),
                threshold=threshold,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                verbose=verbose
            )
            
            # Convert numpy types
            result = convert_numpy_types(result)
            
            # Add audio file info
            result['audio_file'] = str(audio_file)
            result['audio_metadata'] = {
                'num_channels': info.channels,
                'sample_rate_hz': info.samplerate,
                'duration_sec': round(info.duration, 3),
                'bit_depth': info.subtype
            }
            
            # Remove channel_data from output (it's large and redundant)
            channel_data = result.pop('channel_data', {})
            
            # Save JSON
            output_json_path = audio_file.parent / f"{audio_file.stem}_diarization_silero_multispeaker.json"
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"    ✓ Saved: {output_json_path.name}")
            
            # Print summary
            print(f"\n    Multi-Speaker Detection Results:")
            print_speaker_summary(channel_data)
            
            print(f"\n    Overall Metrics:")
            print_diarization_summary(result)
            
            processed_count += 1
            
        except Exception as e:
            print(f"    ✗ Error processing {audio_file.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            error_count += 1
    
    return processed_count, error_count


def main():
    """
    Main function to perform multi-speaker diarization.
    """
    parser = argparse.ArgumentParser(
        description="Multi-speaker diarization using Silero VAD + Speaker Embeddings"
    )
    parser.add_argument("input_path", type=str, 
                       help="Path to audio file or folder containing audio files/subfolders")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="VAD threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--min-speech", type=int, default=250,
                       help="Minimum speech duration in ms (default: 250)")
    parser.add_argument("--min-silence", type=int, default=100,
                       help="Minimum silence duration in ms (default: 100)")
    parser.add_argument("--recursive", action="store_true",
                       help="Process each subfolder separately")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Path not found at {input_path.absolute()}")
        return
    
    # Single file mode
    if input_path.is_file():
        print(f"\n{'='*80}")
        print(f"SILERO VAD + SPEAKER EMBEDDINGS MULTI-SPEAKER DIARIZATION")
        print(f"Target File: {input_path.absolute()}")
        print(f"Parameters: threshold={args.threshold}, min_speech={args.min_speech}ms, min_silence={args.min_silence}ms")
        print(f"{'='*80}")
        
        # Process single file
        folder_path = input_path.parent
        
        # Get audio metadata
        info = sf.info(str(input_path))
        print(f"\nAudio Info:")
        print(f"  Channels: {info.channels}")
        print(f"  Sample Rate: {info.samplerate} Hz")
        print(f"  Duration: {info.duration:.2f}s")
        print(f"  Bit Depth: {info.subtype}")
        
        print(f"\nProcessing...")
        
        try:
            # Generate diarization
            result = generate_diarization_silero_multispeaker(
                str(input_path),
                threshold=args.threshold,
                min_speech_duration_ms=args.min_speech,
                min_silence_duration_ms=args.min_silence,
                verbose=args.verbose
            )
            
            # Convert numpy types
            result = convert_numpy_types(result)
            
            # Add audio file info
            result['audio_file'] = str(input_path)
            result['audio_metadata'] = {
                'num_channels': info.channels,
                'sample_rate_hz': info.samplerate,
                'duration_sec': round(info.duration, 3),
                'bit_depth': info.subtype
            }
            
            # Remove channel_data from output
            channel_data = result.pop('channel_data', {})
            
            # Save JSON
            output_json_path = input_path.parent / f"{input_path.stem}_diarization_silero_multispeaker.json"
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Saved: {output_json_path}")
            
            # Print summary
            print(f"\n{'='*80}")
            print("MULTI-SPEAKER DETECTION RESULTS")
            print(f"{'='*80}")
            print_speaker_summary(channel_data)
            
            print(f"\n{'='*80}")
            print("OVERALL METRICS")
            print(f"{'='*80}")
            print_diarization_summary(result)
            
            processed_count = 1
            error_count = 0
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            processed_count = 0
            error_count = 1
    
    # Folder mode
    elif input_path.is_dir():
        # Recursive subfolder mode
        if args.recursive:
            print(f"\n{'='*80}")
            print(f"SILERO VAD + SPEAKER EMBEDDINGS - RECURSIVE SUBFOLDER MODE")
            print(f"Parent Folder: {input_path.absolute()}")
            print(f"Parameters: threshold={args.threshold}, min_speech={args.min_speech}ms, min_silence={args.min_silence}ms")
            print(f"{'='*80}")
            
            # Find all subfolders
            subfolders = find_subfolders(input_path)
            
            if not subfolders:
                print(f"\nNo subfolders found in {input_path}")
                print(f"Tip: Use without --recursive to process files directly in this folder")
                return
            
            print(f"\nFound {len(subfolders)} subfolder(s) to process")
            
            # Process each subfolder
            total_processed = 0
            total_errors = 0
            
            for idx, subfolder in enumerate(subfolders, 1):
                print(f"\n{'='*80}")
                print(f"[SUBFOLDER {idx}/{len(subfolders)}] Processing: {subfolder.name}")
                print(f"{'='*80}")
                
                processed, errors = process_folder(
                    subfolder, 
                    args.threshold, 
                    args.min_speech, 
                    args.min_silence,
                    args.verbose
                )
                
                total_processed += processed
                total_errors += errors
                
                print(f"\n  Subfolder Summary:")
                print(f"    Files processed: {processed}")
                print(f"    Errors: {errors}")
            
            processed_count = total_processed
            error_count = total_errors
            
        # Non-recursive folder mode
        else:
            print(f"\n{'='*80}")
            print(f"SILERO VAD + SPEAKER EMBEDDINGS - FOLDER MODE")
            print(f"Target Folder: {input_path.absolute()}")
            print(f"Parameters: threshold={args.threshold}, min_speech={args.min_speech}ms, min_silence={args.min_silence}ms")
            print(f"{'='*80}")
            
            processed_count, error_count = process_folder(
                input_path, 
                args.threshold, 
                args.min_speech, 
                args.min_silence,
                args.verbose
            )
    
    else:
        print(f"Error: {input_path.absolute()} is not a valid file or directory")
        return
    
    # Print final summary
    print(f"\n{'='*80}")
    print("DIARIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {processed_count}")
    print(f"Total errors encountered: {error_count}")
    
    if processed_count > 0:
        success_rate = (processed_count / (processed_count + error_count) * 100)
        print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\n✓ Multi-speaker diarization completed!")
        print(f"\nOutput files: *_diarization_silero_multispeaker.json")
        print(f"\nJSON structure includes:")
        print(f"  ├── method & description")
        print(f"  ├── parameters used")
        print(f"  ├── speech_segments (with speaker labels per channel)")
        print(f"  ├── silence_segments (all, leading, trailing, middle)")
        print(f"  ├── diarization_metrics (balance, naturalness, etc.)")
        print(f"  ├── multispeaker_metadata:")
        print(f"  │   ├── multispeaker_likelihood (bool)")
        print(f"  │   ├── potential_number_of_speakers_in_channel (int)")
        print(f"  │   └── speech_ratio_for_all_speakers_across_channel (dict)")
        print(f"  └── audio_metadata")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()