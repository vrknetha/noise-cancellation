#!/usr/bin/env python3
"""
Simple Noise Reducer Test Script
================================

This script helps test and fine-tune the noise_reducer.py parameters.
Use this to find the best settings for your audio environment.

Usage:
    python3 test_noise_reducer.py                    # Test with default settings
    python3 test_noise_reducer.py --preset gentle    # Test with preset
    python3 test_noise_reducer.py --custom           # Test with custom settings
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import time
from noise_reducer import EnhancedNoiseReducer

def load_audio(filename="WhatsApp Audio 2025-07-11 at 12.30.43 PM.wav"):
    """Load and preprocess audio file"""
    print(f"Loading: {filename}")
    
    audio, sr = sf.read(filename)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        sr = 16000
    
    # Convert to int16
    audio = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    
    print(f"Audio: {len(audio)/sr:.1f}s at {sr}Hz")
    return audio, sr

def test_noise_reducer(audio, sr, reducer, name="test"):
    """Test noise reducer and return metrics"""
    print(f"\n=== Testing {name} ===")
    
    # Process audio in chunks
    chunk_size = 160  # 10ms at 16kHz
    processed_audio = []
    latencies = []
    
    start_time = time.time()
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        processed_chunk, metrics = reducer.process_chunk(chunk)
        processed_audio.extend(processed_chunk[:min(len(chunk), chunk_size)])
        
        if 'latency_ms' in metrics:
            latencies.append(metrics['latency_ms'])
    
    total_time = time.time() - start_time
    processed_audio = np.array(processed_audio[:len(audio)])
    
    # Calculate quality metrics
    correlation = np.corrcoef(audio, processed_audio)[0, 1]
    rms_original = np.sqrt(np.mean(audio**2))
    rms_processed = np.sqrt(np.mean(processed_audio**2))
    rms_reduction = 1 - (rms_processed / rms_original)
    avg_latency = np.mean(latencies) if latencies else 0
    
    # Save output
    output_filename = f"output_{name}.wav"
    sf.write(output_filename, processed_audio, sr)
    
    # Print results
    print(f"Results:")
    print(f"  Voice Quality: {correlation:.3f} (target: >0.85)")
    print(f"  Noise Reduction: {rms_reduction*100:.1f}%")
    print(f"  Avg Latency: {avg_latency:.2f}ms (target: <20ms)")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Saved: {output_filename}")
    
    return {
        'name': name,
        'audio': processed_audio,
        'correlation': correlation,
        'rms_reduction': rms_reduction,
        'avg_latency': avg_latency,
        'total_time': total_time,
        'filename': output_filename
    }

def apply_preset(reducer, preset):
    """Apply predefined parameter preset"""
    presets = {
        'optimal': {
            'prop_decrease': 0.6,
            'time_constant_s': 0.15,
            'freq_mask_smooth_hz': 250,
            'highpass_cutoff': 80,
            'noise_gate_threshold': -45,
            'compressor_ratio': 3.0,
            'telephony_band': True,
            'presence_boost': 2.0
        },
        'gentle': {
            'prop_decrease': 0.5,
            'time_constant_s': 0.4,
            'freq_mask_smooth_hz': 200,
            'highpass_cutoff': 60,
            'noise_gate_threshold': -40,
            'compressor_ratio': 2.0,
            'telephony_band': False,
            'presence_boost': 1.0
        },
        'standard': {
            'prop_decrease': 0.7,
            'time_constant_s': 0.3,
            'freq_mask_smooth_hz': 300,
            'highpass_cutoff': 80,
            'noise_gate_threshold': -35,
            'compressor_ratio': 3.0,
            'telephony_band': False,
            'presence_boost': 2.0
        },
        'aggressive': {
            'prop_decrease': 0.85,
            'time_constant_s': 0.2,
            'freq_mask_smooth_hz': 350,
            'highpass_cutoff': 100,
            'noise_gate_threshold': -30,
            'compressor_ratio': 4.0,
            'telephony_band': True,
            'presence_boost': 3.0
        }
    }
    
    if preset not in presets:
        print(f"Unknown preset: {preset}")
        print(f"Available: {list(presets.keys())}")
        return
    
    settings = presets[preset]
    
    # Update reducer parameters
    reducer.prop_decrease = settings['prop_decrease']
    reducer.time_constant_s = settings['time_constant_s']
    reducer.freq_mask_smooth_hz = settings['freq_mask_smooth_hz']
    
    # Update pedalboard based on preset
    from pedalboard import Pedalboard, NoiseGate, Compressor, HighpassFilter, LowpassFilter, PeakFilter, Limiter
    
    board_effects = [
        # Stage 1: Remove handling noise
        HighpassFilter(cutoff_frequency_hz=settings['highpass_cutoff']),
        
        # Stage 2: Noise gate
        NoiseGate(
            threshold_db=settings['noise_gate_threshold'],
            ratio=10,
            attack_ms=1,
            release_ms=50
        ),
        
        # Stage 3: Compressor
        Compressor(
            threshold_db=-20,
            ratio=settings['compressor_ratio'],
            attack_ms=5,
            release_ms=80
        )
    ]
    
    # Add telephony band filtering if enabled
    if settings['telephony_band']:
        board_effects.extend([
            HighpassFilter(cutoff_frequency_hz=300),     # Telephony band
            LowpassFilter(cutoff_frequency_hz=3400),     # Telephony band
        ])
    
    # Add presence boost and limiter
    board_effects.extend([
        PeakFilter(
            cutoff_frequency_hz=2000,
            gain_db=settings['presence_boost'],
            q=1.5
        ),
        Limiter(threshold_db=-3, release_ms=50)
    ])
    
    reducer.board = Pedalboard(board_effects)
    
    print(f"Applied preset: {preset}")
    for key, value in settings.items():
        print(f"  {key}: {value}")

def custom_settings():
    """Interactive custom settings"""
    print("\n=== Custom Settings ===")
    
    # Get user input for key parameters
    try:
        prop_decrease = float(input("Noise reduction strength (0.5-0.95, default 0.7): ") or "0.7")
        prop_decrease = max(0.5, min(0.95, prop_decrease))
        
        highpass_cutoff = float(input("High-pass filter cutoff Hz (60-150, default 80): ") or "80")
        highpass_cutoff = max(60, min(150, highpass_cutoff))
        
        noise_gate_threshold = float(input("Noise gate threshold dB (-50 to -25, default -35): ") or "-35")
        noise_gate_threshold = max(-50, min(-25, noise_gate_threshold))
        
        compressor_ratio = float(input("Compressor ratio (2-6, default 3): ") or "3")
        compressor_ratio = max(2, min(6, compressor_ratio))
        
        final_gain = float(input("Final gain dB (0-5, default 2): ") or "2")
        final_gain = max(0, min(5, final_gain))
        
        return {
            'prop_decrease': prop_decrease,
            'highpass_cutoff': highpass_cutoff,
            'noise_gate_threshold': noise_gate_threshold,
            'compressor_ratio': compressor_ratio,
            'final_gain': final_gain
        }
        
    except (ValueError, KeyboardInterrupt):
        print("Using default settings")
        return None

def create_visualization(original, results):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain comparison (first 5 seconds)
    duration = min(5, len(original) / 16000)
    samples = int(duration * 16000)
    time_axis = np.arange(samples) / 16000
    
    axes[0, 0].plot(time_axis, original[:samples], label='Original', alpha=0.7)
    for result in results[:2]:  # Show first 2 results
        axes[0, 0].plot(time_axis, result['audio'][:samples], 
                       label=result['name'], alpha=0.7)
    axes[0, 0].set_title('Waveform Comparison')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Quality metrics
    names = [r['name'] for r in results]
    correlations = [r['correlation'] for r in results]
    rms_reductions = [r['rms_reduction'] * 100 for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, correlations, width, label='Voice Quality', alpha=0.8)
    axes[0, 1].bar(x + width/2, [r/100 for r in rms_reductions], width, label='Noise Reduction', alpha=0.8)
    axes[0, 1].set_title('Quality Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Latency comparison
    latencies = [r['avg_latency'] for r in results]
    axes[1, 0].bar(names, latencies, alpha=0.8)
    axes[1, 0].set_title('Average Latency')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Target: 20ms')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frequency comparison
    fft_orig = np.abs(np.fft.rfft(original))
    freqs = np.fft.rfftfreq(len(original), 1/16000)
    
    axes[1, 1].plot(freqs, 20*np.log10(fft_orig + 1e-10), label='Original', alpha=0.7)
    for result in results[:2]:
        fft_processed = np.abs(np.fft.rfft(result['audio']))
        axes[1, 1].plot(freqs, 20*np.log10(fft_processed + 1e-10), 
                       label=result['name'], alpha=0.7)
    axes[1, 1].set_title('Frequency Response')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].set_xlim(0, 4000)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_reduction_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved: noise_reduction_test_results.png")

def main():
    parser = argparse.ArgumentParser(description='Test Noise Reducer')
    parser.add_argument('--preset', choices=['optimal', 'gentle', 'standard', 'aggressive'], 
                       help='Use preset configuration')
    parser.add_argument('--custom', action='store_true', help='Use custom settings')
    parser.add_argument('--input', default="WhatsApp Audio 2025-07-11 at 12.30.43 PM.wav", 
                       help='Input audio file')
    parser.add_argument('--compare', action='store_true', help='Compare multiple presets')
    
    args = parser.parse_args()
    
    # Load audio
    audio, sr = load_audio(args.input)
    
    results = []
    
    if args.compare:
        # Compare all presets
        print("\n=== COMPARING ALL PRESETS ===")
        for preset in ['optimal', 'gentle', 'standard', 'aggressive']:
            reducer = EnhancedNoiseReducer(sr)
            apply_preset(reducer, preset)
            result = test_noise_reducer(audio, sr, reducer, preset)
            results.append(result)
    
    elif args.custom:
        # Custom settings
        settings = custom_settings()
        if settings:
            reducer = EnhancedNoiseReducer(sr)
            apply_preset(reducer, 'standard')  # Start with standard
            
            # Apply custom settings
            reducer.prop_decrease = settings['prop_decrease']
            # Update other parameters as needed
            
            result = test_noise_reducer(audio, sr, reducer, 'custom')
            results.append(result)
    
    elif args.preset:
        # Specific preset
        reducer = EnhancedNoiseReducer(sr)
        apply_preset(reducer, args.preset)
        result = test_noise_reducer(audio, sr, reducer, args.preset)
        results.append(result)
    
    else:
        # Default test - use optimal preset
        reducer = EnhancedNoiseReducer(sr)
        result = test_noise_reducer(audio, sr, reducer, 'optimal_default')
        results.append(result)
    
    # Show results summary
    if results:
        print(f"\n{'='*50}")
        print("RESULTS SUMMARY")
        print(f"{'='*50}")
        
        print(f"{'Config':<12} {'Voice Quality':<14} {'Noise Reduction':<16} {'Latency':<12}")
        print("-" * 55)
        
        for result in results:
            print(f"{result['name']:<12} {result['correlation']:<14.3f} "
                  f"{result['rms_reduction']*100:<16.1f}% {result['avg_latency']:<12.2f}ms")
        
        # Create visualization if multiple results
        if len(results) > 1:
            create_visualization(audio, results)
        
        print(f"\n📁 Output files:")
        for result in results:
            print(f"  {result['filename']}")
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()