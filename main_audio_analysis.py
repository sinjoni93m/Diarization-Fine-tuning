"""
Pure Audio Content Analysis Module

Takes a multichannel audio file and optional language_hint as input and performs:
- Basic pure audio analysis on each channel
- Speech/silence diarization with quality metrics
- Speech-to-text transcription on speech segments (if language is supported)

Generates a single comprehensive JSON file with all results.

Usage: 
    # Analyze single audio file (no transcription):
    python3 main_audio_analysis.py audio.wav 

    # Analyze with transcription (auto-detect language):
    python3 main_audio_analysis.py audio.wav --transcribe

    # Analyze with language hint:
    python3 main_audio_analysis.py audio.wav --transcribe --language en
    python3 main_audio_analysis.py audio.wav --transcribe --language es-MX

    # Analyze all audio files in folder:
    python3 main_audio_analysis.py /path/to/audio/folder --transcribe

    # Verbose analysis:
    python3 main_audio_analysis.py /path/to/audio/folder --transcribe --verbose
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

from pure_audio_analysis import AudioAnalysis
from diarization import AudioDiarization
from transcript_generator import MultichannelTranscriptGenerator

AUDIO_ANALYSIS_VERSION = "3.3.0"


def analyze_audio_file(analyzer: AudioAnalysis, audio_file_path: str) -> Dict[str, Any]:
    """
    Main analysis function that takes an AudioAnalysis instance and:
    - Performs basic audio analysis on every channel of the multichannel file
    - Performs an aggregated analysis for overall audio
    """
    try:
        # Get audio file metadata
        waveform, sample_rate = torchaudio.load(audio_file_path)
        num_channels = waveform.shape[0] if waveform.ndim > 1 else 1
        info = sf.info(audio_file_path)
        
        # Analyze each channel
        channel_results = {}
        for ch_idx in range(analyzer.num_channels):
            channel_data = analyzer.audio[ch_idx]
            channel_analysis = analyzer.basic_audio_metrics_per_channel(ch_idx, channel_data)
            channel_results[ch_idx] = channel_analysis
        
        # Calculate overall audio analysis from channel analyses
        overall_audio_analysis = analyzer.calculate_overall_audio_analysis(channel_results)
        
        # Add file metadata to overall_audio_analysis
        overall_audio_analysis["num_channels"] = num_channels
        overall_audio_analysis["bit_depth"] = info.subtype
        overall_audio_analysis["sample_rate_hz"] = analyzer.sr
        
        # Results structure
        results = {
            "overall_audio_analysis": overall_audio_analysis,
            "channel_analyses": channel_results
        }
        
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def generate_diarization(diarizer: AudioDiarization) -> Dict[str, Any]:
    """
    Generate diarization output with speech segments, silence segments, and quality metrics.
    """
    try:
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
            
            # Add all silence segments with channel info
            for segment in channel_info["silence_segments"]:
                all_silence_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "channel": ch_idx
                })
            
            # Add leading silence segments
            for segment in channel_info.get("leading_silence_segments", []):
                all_leading_silence_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "channel": ch_idx
                })
            
            # Add trailing silence segments
            for segment in channel_info.get("trailing_silence_segments", []):
                all_trailing_silence_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "channel": ch_idx
                })
            
            # Add middle silence segments
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
            "speech_segments": all_speech_segments,
            "all_silence_segments": all_silence_segments,
            "leading_silence_segments": all_leading_silence_segments,
            "trailing_silence_segments": all_trailing_silence_segments,
            "middle_silence_segments": all_middle_silence_segments,
            "diarization_metrics": metrics,
            "num_channels": diarizer.num_channels,
            "sample_rate_hz": diarizer.sr,
            "channel_data": channel_data  # Include for transcription
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def generate_transcript(generator: MultichannelTranscriptGenerator,
                       diarization_result: Dict[str, Any],
                       language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate transcript from diarization output for speech segments only.
    """
    try:
        # Check if language hint is supported (if provided)
        if language_hint and not generator.is_language_supported(language_hint):
            base_lang = generator._get_base_language_code(language_hint)
            return {
                "error": f"Language '{language_hint}' (base: '{base_lang}') is not supported by Whisper",
                "primary_language": None,
                "detected_languages": [],
                "language_hint_provided": True,
                "language_hint": language_hint,
                "transcript_segments": []
            }
        
        # Transcribe speech segments from diarization (returns dict with new structure)
        transcript_result = generator.transcribe_speech_segments_from_diarization(diarization_result)
        
        # Sort transcript segments by start time
        if transcript_result.get("transcript_segments"):
            transcript_result["transcript_segments"].sort(key=lambda x: x["start"])
        
        return transcript_result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "primary_language": None,
            "detected_languages": [],
            "language_hint_provided": language_hint is not None,
            "language_hint": language_hint,
            "transcript_segments": []
        }


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
    Convert numpy types to serializable types
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    return obj


def build_comprehensive_json(audio_file_path: str, 
                             analysis_result: Dict[str, Any],
                             diarization_result: Dict[str, Any],
                             transcript_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a single comprehensive JSON with all analysis results.
    """
    # Remove channel_data from diarization (internal use only)
    diarization_clean = {k: v for k, v in diarization_result.items() if k != "channel_data"}
    
    # Build comprehensive result
    comprehensive = {
        "audio_file": audio_file_path,
        "overall_audio_analysis": analysis_result.get("overall_audio_analysis", {}),
        "channel_analyses": analysis_result.get("channel_analyses", {}),
        "diarization": diarization_clean,
    }
    
    # Add transcript if available and has segments
    if transcript_result and transcript_result.get("transcript_segments"):
        comprehensive["transcript"] = transcript_result
    
    return comprehensive


def main_audio_analysis():
    """
    Main function to perform comprehensive audio analysis on audio file(s).
    Generates a single comprehensive JSON file per audio file with:
    - Audio analysis (per-channel and overall metrics)
    - Diarization (speech/silence segments with quality metrics)
    - Transcript (speech-to-text if --transcribe flag is used and language supported)
    """
    parser = argparse.ArgumentParser(
        description="Analyze audio files with per-channel analysis, diarization, and optional transcription"
    )
    parser.add_argument("input_path", type=str, help="Path to audio file or folder containing audio files")
    parser.add_argument("--transcribe", action="store_true",
                       help="Enable speech-to-text transcription on speech segments")
    parser.add_argument("--language", type=str, default=None,
                       help="Language hint for transcription (e.g., 'en', 'es', 'en-US'). If not provided, auto-detect.")
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
    
    # Determine if input is file or folder
    if input_path.is_file():
        # Single file mode
        audio_files = [input_path]
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE AUDIO ANALYSIS - SINGLE FILE")
        print(f"Target File: {input_path.absolute()}")
        if args.transcribe:
            print(f"Transcription: Enabled")
            print(f"Language: {args.language if args.language else 'Auto-detect'}")
        print(f"{'='*80}")
    elif input_path.is_dir():
        # Folder mode
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE AUDIO ANALYSIS - FOLDER MODE")
        print(f"Target Folder: {input_path.absolute()}")
        if args.transcribe:
            print(f"Transcription: Enabled")
            print(f"Language: {args.language if args.language else 'Auto-detect'}")
        print(f"{'='*80}")
        
        # Find all audio files
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
    transcription_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Analyzing: {audio_file.name}")
        
        try:
            # Create audio analyzer and diarizer
            analyzer = AudioAnalysis(str(audio_file))
            diarizer = AudioDiarization(str(audio_file))
            
            # Perform audio analysis
            print(f"  Running audio analysis...")
            analysis_result = analyze_audio_file(analyzer, str(audio_file))
            analysis_result = convert_numpy_types(analysis_result)
            
            # Perform diarization
            print(f"  Running diarization...")
            diarization_result = generate_diarization(diarizer)
            diarization_result = convert_numpy_types(diarization_result)
            
            # Perform transcription if requested
            transcript_result = None
            if args.transcribe:
                generator = MultichannelTranscriptGenerator(
                    str(audio_file), 
                    language_hint=args.language
                )
                
                # Check if language is supported (if hint provided)
                if args.language and not generator.is_language_supported(args.language):
                    base_lang = generator._get_base_language_code(args.language)
                    print(f"  ⚠ Language '{args.language}' (base: '{base_lang}') not supported by Whisper. Skipping transcription.")
                else:
                    print(f"  Running transcription...")
                    transcript_result = generate_transcript(
                        generator, 
                        diarization_result, 
                        args.language
                    )
                    transcript_result = convert_numpy_types(transcript_result)
                    
                    # Check if transcription was successful (has segments)
                    if transcript_result.get("transcript_segments"):
                        transcription_count += 1
                        
                        # Print transcript summary
                        print(f"\n  Transcription Summary:")
                        print(f"    Language hint: {transcript_result.get('language_hint', 'None (auto-detect)')}")
                        print(f"    Primary language: {transcript_result.get('primary_language', 'unknown')}")
                        print(f"    Detected languages: {', '.join(transcript_result.get('detected_languages', []))}")
                        print(f"    Total segments: {len(transcript_result.get('transcript_segments', []))}")
            
            # Build comprehensive JSON
            comprehensive_result = build_comprehensive_json(
                str(audio_file),
                analysis_result,
                diarization_result,
                transcript_result
            )
            
            # Generate output JSON path
            output_json_path = audio_file.parent / f"{audio_file.stem}_comprehensive_analysis.json"
            
            # Save comprehensive JSON
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_result, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Comprehensive analysis saved: {output_json_path.name}")
            
            # Print audio analysis summary
            if 'overall_audio_analysis' in comprehensive_result:
                overall = comprehensive_result['overall_audio_analysis']
                print(f"\n  Audio Metrics:")
                print(f"    Channels: {overall.get('num_channels', 0)}")
                print(f"    Sample Rate: {overall.get('sample_rate_hz', 0)} Hz")
                print(f"    Avg SNR: {overall.get('avg_snr_db', 0):.2f} dB")
                print(f"    Avg RMS Volume: {overall.get('avg_rms_volume', 0):.6f}")
                print(f"    Clipping Detected: {overall.get('any_clipped', False)}")
            
            # Print diarization summary
            if 'diarization' in comprehensive_result:
                diarization = comprehensive_result['diarization']
                metrics = diarization.get('diarization_metrics', {})
                num_speech = len(diarization.get('speech_segments', []))
                num_silence = len(diarization.get('all_silence_segments', []))
                num_leading = len(diarization.get('leading_silence_segments', []))
                num_trailing = len(diarization.get('trailing_silence_segments', []))
                num_middle = len(diarization.get('middle_silence_segments', []))
                
                print(f"\n  Diarization Metrics:")
                print(f"    Speech segments: {num_speech}")
                print(f"    Silence segments: {num_silence}")
                print(f"      - Leading: {num_leading} ({metrics.get('avg_leading_silence_sec', 0):.2f}s avg)")
                print(f"      - Middle: {num_middle} ({metrics.get('avg_middle_silence_sec', 0):.2f}s avg)")
                print(f"      - Trailing: {num_trailing} ({metrics.get('avg_trailing_silence_sec', 0):.2f}s avg)")
                print(f"    Balance: {metrics.get('balance_score', 0):.3f} ({metrics.get('balance_assessment', 'N/A')})")
                print(f"    Naturalness: {metrics.get('naturalness_score', 0):.3f}")
                print(f"    Avg silence: {metrics.get('avg_silence_percentage', 0):.1f}%")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error analyzing {audio_file.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            error_count += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total audio files found: {len(audio_files)}")
    print(f"Successfully analyzed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    
    if args.transcribe:
        print(f"Transcriptions generated: {transcription_count}")
    
    if processed_count > 0:
        print(f"\nAnalysis completed successfully!")
        print(f"Generated comprehensive JSON files:")
        print(f"  - *_comprehensive_analysis.json")
        print(f"      ├── overall_audio_analysis")
        print(f"      ├── channel_analyses")
        print(f"      ├── diarization")
        if args.transcribe:
            print(f"      └── transcript (if language supported)")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main_audio_analysis()