"""
Multi-Method Diarization Comparison Module

Takes a multichannel audio file as input and performs speech/silence diarization 
using FOUR different methods:
1. Energy-Based Detection (legacy, simple)
2. Spectral Features Detection (robust, adaptive)
3. WebRTC VAD (industry-standard)
4. Silero VAD (deep learning, state-of-the-art)

Generates separate JSON files for each method with comprehensive results.

Usage: 
    # Analyze single audio file with all methods:
    python3 main_diarization_comparison.py audio.wav
    
    # Analyze folder with all methods:
    python3 main_diarization_comparison.py /path/to/audio/folder
    
    # Analyze with specific methods only:
    python3 main_diarization_comparison.py audio.wav --methods energy webrtc
    
    # Verbose output:
    python3 main_diarization_comparison.py audio.wav --verbose
"""

import os
import argparse
import logging
import numpy as np
import soundfile as sf
import json
from typing import Dict, Any, List, Optional
import torchaudio
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import all diarization methods
from diarization_old import AudioDiarization as AudioDiarizationEnergy
from diarization_spectral_adaptive import AudioDiarization as AudioDiarizationSpectral
from diarization_webrtc_vad import AudioDiarization as AudioDiarizationWebRTC
from diarization_silero import AudioDiarization as AudioDiarizationSilero


def generate_diarization_energy(audio_file_path: str) -> Dict[str, Any]:
    """
    Generate diarization using Energy-Based Detection (legacy method).
    """
    try:
        print("    Method: Energy-Based Detection")
        print("    - Simple energy thresholding")
        print("    - Fast but noise-sensitive")
        
        diarizer = AudioDiarizationEnergy(audio_file_path)
        
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
            
            # Add speech segments with speaker_id
            for segment in channel_info["speech_segments"]:
                all_speech_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": f"channel_{ch_idx}"
                })
            
            # Add all silence segments
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
        
        # Sort segments by start time
        all_speech_segments.sort(key=lambda x: x["start"])
        all_silence_segments.sort(key=lambda x: x["start"])
        all_leading_silence_segments.sort(key=lambda x: x["start"])
        all_trailing_silence_segments.sort(key=lambda x: x["start"])
        all_middle_silence_segments.sort(key=lambda x: x["start"])
        
        # Calculate metrics
        metrics = diarizer.calculate_metrics(channel_data)
        
        return {
            "method": "energy_based",
            "method_description": "Simple energy-based detection with fixed dB threshold",
            "parameters": {
                "silence_thresh_db": -40,
                "min_speech_duration": 0.3,
                "min_silence_duration": 0.3
            },
            "speech_segments": all_speech_segments,
            "all_silence_segments": all_silence_segments,
            "leading_silence_segments": all_leading_silence_segments,
            "trailing_silence_segments": all_trailing_silence_segments,
            "middle_silence_segments": all_middle_silence_segments,
            "diarization_metrics": metrics,
            "num_channels": diarizer.num_channels,
            "sample_rate_hz": diarizer.sr
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "method": "energy_based"}


def generate_diarization_spectral(audio_file_path: str, percentile_threshold: int = 30) -> Dict[str, Any]:
    """
    Generate diarization using Spectral Features Detection (adaptive method).
    """
    try:
        print("    Method: Spectral Features Detection")
        print("    - Uses spectral centroid + rolloff")
        print("    - Adaptive thresholding (noise-robust)")
        
        diarizer = AudioDiarizationSpectral(audio_file_path)
        
        all_speech_segments = []
        all_silence_segments = []
        all_leading_silence_segments = []
        all_trailing_silence_segments = []
        all_middle_silence_segments = []
        channel_data = {}
        
        # Extract segments from each channel
        for ch_idx in range(diarizer.num_channels):
            channel_audio = diarizer.audio[ch_idx]
            channel_info = diarizer.extract_channel_segments(
                ch_idx, 
                channel_audio, 
                percentile_threshold=percentile_threshold
            )
            channel_data[ch_idx] = channel_info
            
            # Add speech segments
            for segment in channel_info["speech_segments"]:
                all_speech_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": f"channel_{ch_idx}"
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
        
        return {
            "method": "spectral_features",
            "method_description": "Spectral features (centroid + rolloff) with adaptive thresholding",
            "parameters": {
                "percentile_threshold": percentile_threshold,
                "hop_length": 512,
                "min_speech_duration": 0.3,
                "min_silence_duration": 0.3
            },
            "speech_segments": all_speech_segments,
            "all_silence_segments": all_silence_segments,
            "leading_silence_segments": all_leading_silence_segments,
            "trailing_silence_segments": all_trailing_silence_segments,
            "middle_silence_segments": all_middle_silence_segments,
            "diarization_metrics": metrics,
            "num_channels": diarizer.num_channels,
            "sample_rate_hz": diarizer.sr
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "method": "spectral_features"}


def generate_diarization_webrtc(audio_file_path: str, vad_aggressiveness: int = 2) -> Dict[str, Any]:
    """
    Generate diarization using WebRTC VAD (industry-standard method).
    """
    try:
        print("    Method: WebRTC VAD")
        print("    - Industry-standard voice activity detection")
        print(f"    - Aggressiveness: {vad_aggressiveness} (0=quality, 3=very aggressive)")
        
        diarizer = AudioDiarizationWebRTC(audio_file_path, vad_aggressiveness=vad_aggressiveness)
        
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
            
            # Add speech segments
            for segment in channel_info["speech_segments"]:
                all_speech_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": f"channel_{ch_idx}"
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
        
        return {
            "method": "webrtc_vad",
            "method_description": "WebRTC Voice Activity Detection (GMM-based)",
            "parameters": {
                "vad_aggressiveness": vad_aggressiveness,
                "frame_duration_ms": 30,
                "min_speech_duration": 0.3,
                "min_silence_duration": 0.3
            },
            "speech_segments": all_speech_segments,
            "all_silence_segments": all_silence_segments,
            "leading_silence_segments": all_leading_silence_segments,
            "trailing_silence_segments": all_trailing_silence_segments,
            "middle_silence_segments": all_middle_silence_segments,
            "diarization_metrics": metrics,
            "num_channels": diarizer.num_channels,
            "sample_rate_hz": diarizer.sr
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "method": "webrtc_vad"}


def generate_diarization_silero(audio_file_path: str, 
                                threshold: float = 0.5,
                                min_speech_duration_ms: int = 250,
                                min_silence_duration_ms: int = 100) -> Dict[str, Any]:
    """
    Generate diarization using Silero VAD (deep learning method).
    """
    try:
        print("    Method: Silero VAD")
        print("    - Deep learning (neural network)")
        print("    - State-of-the-art accuracy")
        
        diarizer = AudioDiarizationSilero(
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
            
            # Add speech segments
            for segment in channel_info["speech_segments"]:
                all_speech_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": f"channel_{ch_idx}"
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
        
        return {
            "method": "silero_vad",
            "method_description": "Silero VAD - Deep Learning Neural Network",
            "parameters": {
                "threshold": threshold,
                "min_speech_duration_ms": min_speech_duration_ms,
                "min_silence_duration_ms": min_silence_duration_ms,
                "window_size_samples": 512,
                "speech_pad_ms": 30
            },
            "speech_segments": all_speech_segments,
            "all_silence_segments": all_silence_segments,
            "leading_silence_segments": all_leading_silence_segments,
            "trailing_silence_segments": all_trailing_silence_segments,
            "middle_silence_segments": all_middle_silence_segments,
            "diarization_metrics": metrics,
            "num_channels": diarizer.num_channels,
            "sample_rate_hz": diarizer.sr
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "method": "silero_vad"}


def find_audio_files(folder_path: Path) -> List[Path]:
    """
    Find all audio files in the given folder
    """
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.aiff', '.au')):
                audio_files.append(Path(root) / file)
    return audio_files


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


def main():
    """
    Main function to perform multi-method diarization comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare multiple diarization methods on audio files"
    )
    parser.add_argument("input_path", type=str, 
                       help="Path to audio file or folder containing audio files")
    parser.add_argument("--methods", nargs='+', 
                       choices=['energy', 'spectral', 'webrtc', 'silero', 'all'],
                       default=['all'],
                       help="Methods to use (default: all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Determine which methods to use
    if 'all' in args.methods:
        methods_to_use = ['energy', 'spectral', 'webrtc', 'silero']
    else:
        methods_to_use = args.methods
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Path not found at {input_path.absolute()}")
        return
    
    # Determine if input is file or folder
    if input_path.is_file():
        audio_files = [input_path]
        print(f"\n{'='*80}")
        print(f"MULTI-METHOD DIARIZATION COMPARISON - SINGLE FILE")
        print(f"Target File: {input_path.absolute()}")
        print(f"Methods: {', '.join(methods_to_use)}")
        print(f"{'='*80}")
    elif input_path.is_dir():
        print(f"\n{'='*80}")
        print(f"MULTI-METHOD DIARIZATION COMPARISON - FOLDER MODE")
        print(f"Target Folder: {input_path.absolute()}")
        print(f"Methods: {', '.join(methods_to_use)}")
        print(f"{'='*80}")
        
        audio_files = find_audio_files(input_path)
        if not audio_files:
            print(f"No audio files found in {input_path}")
            return
        
        print(f"Found {len(audio_files)} audio files")
    else:
        print(f"Error: {input_path.absolute()} is not a valid file or directory")
        return
    
    # Process each audio file
    processed_count = 0
    error_count = 0
    method_stats = {method: {'success': 0, 'error': 0} for method in methods_to_use}
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            # Get audio metadata
            info = sf.info(str(audio_file))
            print(f"  Channels: {info.channels}, Sample Rate: {info.samplerate} Hz, Duration: {info.duration:.2f}s")
            
            # Process with each selected method
            for method in methods_to_use:
                print(f"\n  [{method.upper()}] Running diarization...")
                
                try:
                    # Generate diarization based on method
                    if method == 'energy':
                        result = generate_diarization_energy(str(audio_file))
                        output_suffix = "energy_based"
                    elif method == 'spectral':
                        result = generate_diarization_spectral(str(audio_file))
                        output_suffix = "spectral_features"
                    elif method == 'webrtc':
                        result = generate_diarization_webrtc(str(audio_file))
                        output_suffix = "webrtc_vad"
                    elif method == 'silero':
                        result = generate_diarization_silero(str(audio_file))
                        output_suffix = "silero_vad"
                    
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
                    
                    # Save JSON
                    output_json_path = audio_file.parent / f"{audio_file.stem}_diarization_{output_suffix}.json"
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"    ✓ Saved: {output_json_path.name}")
                    
                    # Print summary
                    print_diarization_summary(result)
                    
                    # Update stats
                    if "error" in result:
                        method_stats[method]['error'] += 1
                    else:
                        method_stats[method]['success'] += 1
                    
                except Exception as e:
                    print(f"    ✗ Error in {method}: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    method_stats[method]['error'] += 1
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {audio_file.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            error_count += 1
    
    # Print final summary
    print(f"\n{'='*80}")
    print("DIARIZATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Total audio files found: {len(audio_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"\nMethod Statistics:")
    
    for method in methods_to_use:
        stats = method_stats[method]
        total = stats['success'] + stats['error']
        success_rate = (stats['success'] / total * 100) if total > 0 else 0
        print(f"  {method.upper():12s}: {stats['success']}/{total} successful ({success_rate:.1f}%)")
    
    if processed_count > 0:
        print(f"\n✓ Diarization comparison completed!")
        print(f"\nGenerated JSON files per method:")
        if 'energy' in methods_to_use:
            print(f"  - *_diarization_energy_based.json")
        if 'spectral' in methods_to_use:
            print(f"  - *_diarization_spectral_features.json")
        if 'webrtc' in methods_to_use:
            print(f"  - *_diarization_webrtc_vad.json")
        if 'silero' in methods_to_use:
            print(f"  - *_diarization_silero_vad.json")
        
        print(f"\nEach file contains:")
        print(f"  ├── method & description")
        print(f"  ├── parameters used")
        print(f"  ├── speech_segments (with speaker/channel)")
        print(f"  ├── silence_segments (all, leading, trailing, middle)")
        print(f"  ├── diarization_metrics (balance, naturalness, etc.)")
        print(f"  └── audio_metadata")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()