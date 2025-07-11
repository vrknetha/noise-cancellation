# Audio Noise Reduction Implementation for UKTI Voice Platform

## Project Context
- **Goal**: Implement real-time noise reduction for phone calls via Plivo -> FastRTC -> Gemini Live API pipeline
- **Problem**: Street noise (traffic, crowds) causing Gemini to interrupt callers
- **Audio Format**: PCM 16kHz from Plivo (LINEAR16)
- **Approach**: EnhancedNoiseReducer using noisereduce + Pedalboard
- **Priority**: Balance voice quality preservation with low latency

## Key Technical Requirements
- **Sample Rate**: 16000 Hz (Plivo PCM format)
- **Input Format**: PCM 16-bit signed integer (LINEAR16)
- **Output Format**: PCM 16-bit for Gemini Live API
- **Target Latency**: <20ms per chunk
- **Voice Quality**: >0.9 correlation with original

## Quick Start Implementation

### 1. Install Dependencies
```bash
pip install noisereduce pedalboard numpy scipy soundfile matplotlib
```

### 2. EnhancedNoiseReducer Implementation
```python
import numpy as np
import noisereduce as nr
from pedalboard import Pedalboard, NoiseGate, Compressor, Gain, LowShelfFilter, HighpassFilter
import soundfile as sf
from typing import Tuple, Dict
import time

class EnhancedNoiseReducer:
    """
    Balanced noise reduction for voice calls
    Optimized for 16kHz PCM audio from Plivo
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.frame_count = 0
        self.noise_profile_duration = 0.3  # 300ms for noise profiling
        self.noise_profile_frames = int(self.noise_profile_duration * sample_rate / 160)  # 10ms frames
        
        # Noise reduction parameters (balanced for voice preservation)
        self.prop_decrease = 0.7  # 70% noise reduction (less aggressive)
        self.time_constant_s = 0.3  # Faster adaptation for real-time
        
        # Pedalboard setup for voice enhancement
        self.board = Pedalboard([
            # Remove low-frequency rumble (street noise)
            HighpassFilter(cutoff_frequency_hz=80),
            
            # Gentle noise gate for silence
            NoiseGate(
                threshold_db=-35,  # Less aggressive
                ratio=4,
                attack_ms=1,
                release_ms=50
            ),
            
            # Compress dynamics for consistent voice level
            Compressor(
                threshold_db=-20,
                ratio=3,  # Gentle compression
                attack_ms=1,
                release_ms=50
            ),
            
            # Reduce low-mid frequency mud
            LowShelfFilter(
                cutoff_frequency_hz=250,
                gain_db=-3  # Gentle reduction
            ),
            
            # Slight boost for clarity
            Gain(gain_db=2)
        ])
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a single audio chunk with metrics
        Returns: (processed_audio, metrics)
        """
        start_time = time.time()
        metrics = {}
        
        # Collect noise profile from initial frames
        if self.frame_count < self.noise_profile_frames:
            if self.noise_profile is None:
                self.noise_profile = audio_chunk.copy()
            else:
                # Rolling average for noise profile
                self.noise_profile = (self.noise_profile * 0.9 + audio_chunk * 0.1)
            self.frame_count += 1
            
            # Return lightly processed audio during profiling
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            processed = self.board(audio_float, self.sample_rate)
            processed_int = (processed * 32768).clip(-32768, 32767).astype(np.int16)
            
            metrics['latency_ms'] = (time.time() - start_time) * 1000
            metrics['noise_profiling'] = True
            return processed_int, metrics
        
        # Apply noise reduction
        try:
            # Step 1: Statistical noise reduction
            reduced_audio = nr.reduce_noise(
                y=audio_chunk,
                sr=self.sample_rate,
                y_noise=self.noise_profile,
                prop_decrease=self.prop_decrease,
                stationary=False,
                time_constant_s=self.time_constant_s
            )
            
            # Step 2: Convert to float32 for Pedalboard
            audio_float = reduced_audio.astype(np.float32) / 32768.0
            
            # Step 3: Apply audio enhancement
            processed = self.board(audio_float, self.sample_rate)
            
            # Step 4: Convert back to int16
            processed_int = (processed * 32768).clip(-32768, 32767).astype(np.int16)
            
            # Calculate metrics
            metrics['latency_ms'] = (time.time() - start_time) * 1000
            metrics['rms_before'] = np.sqrt(np.mean(audio_chunk**2))
            metrics['rms_after'] = np.sqrt(np.mean(processed_int**2))
            metrics['noise_reduction_db'] = 20 * np.log10(metrics['rms_after'] / max(metrics['rms_before'], 1e-10))
            
            return processed_int, metrics
            
        except Exception as e:
            print(f"Processing error: {e}")
            metrics['error'] = str(e)
            metrics['latency_ms'] = (time.time() - start_time) * 1000
            return audio_chunk, metrics  # Return original on error
```

### 3. Test Script for WAV File
```python
import matplotlib.pyplot as plt
from scipy import signal

def test_noise_reduction(input_wav_path: str, output_wav_path: str = "output_cleaned.wav"):
    """
    Test noise reduction on a WAV file with visual analysis
    """
    # Load audio
    audio, sr = sf.read(input_wav_path)
    if sr != 16000:
        print(f"Warning: Sample rate is {sr}, expected 16000. Resampling...")
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        sr = 16000
    
    # Convert to int16 if needed
    if audio.dtype != np.int16:
        audio = (audio * 32768).astype(np.int16)
    
    # Initialize reducer
    reducer = EnhancedNoiseReducer(sample_rate=sr)
    
    # Process in chunks (10ms chunks = 160 samples at 16kHz)
    chunk_size = 160
    processed_audio = []
    all_metrics = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        processed_chunk, metrics = reducer.process_chunk(chunk)
        processed_audio.extend(processed_chunk[:min(len(chunk), chunk_size)])
        all_metrics.append(metrics)
    
    processed_audio = np.array(processed_audio[:len(audio)])
    
    # Save output
    sf.write(output_wav_path, processed_audio, sr)
    
    # Calculate overall metrics
    avg_latency = np.mean([m['latency_ms'] for m in all_metrics if 'latency_ms' in m])
    print(f"\nProcessing Results:")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Output saved to: {output_wav_path}")
    
    # Visualize results
    visualize_comparison(audio, processed_audio, sr)
    
    return processed_audio, all_metrics

def visualize_comparison(original: np.ndarray, processed: np.ndarray, sr: int):
    """
    Create visual comparison of before/after
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Time domain
    time = np.arange(len(original)) / sr
    axes[0, 0].plot(time, original)
    axes[0, 0].set_title('Original Waveform')
    axes[0, 0].set_ylabel('Amplitude')
    
    axes[0, 1].plot(time, processed)
    axes[0, 1].set_title('Processed Waveform')
    
    # Spectrograms
    f, t, Sxx_orig = signal.spectrogram(original, sr, nperseg=512)
    f, t, Sxx_proc = signal.spectrogram(processed, sr, nperseg=512)
    
    axes[1, 0].pcolormesh(t, f, 10 * np.log10(Sxx_orig + 1e-10), shading='gouraud')
    axes[1, 0].set_title('Original Spectrogram')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_ylim(0, 4000)  # Focus on voice range
    
    axes[1, 1].pcolormesh(t, f, 10 * np.log10(Sxx_proc + 1e-10), shading='gouraud')
    axes[1, 1].set_title('Processed Spectrogram')
    axes[1, 1].set_ylim(0, 4000)
    
    # Energy comparison
    window_size = int(0.02 * sr)  # 20ms windows
    orig_energy = np.array([np.sqrt(np.mean(original[i:i+window_size]**2)) 
                           for i in range(0, len(original)-window_size, window_size)])
    proc_energy = np.array([np.sqrt(np.mean(processed[i:i+window_size]**2)) 
                           for i in range(0, len(processed)-window_size, window_size)])
    
    energy_time = np.arange(len(orig_energy)) * window_size / sr
    axes[2, 0].plot(energy_time, orig_energy, label='Original')
    axes[2, 0].plot(energy_time, proc_energy, label='Processed')
    axes[2, 0].set_title('Energy Envelope Comparison')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('RMS Energy')
    axes[2, 0].legend()
    
    # Frequency response
    freq_orig = np.fft.rfft(original)
    freq_proc = np.fft.rfft(processed)
    freqs = np.fft.rfftfreq(len(original), 1/sr)
    
    axes[2, 1].plot(freqs, 20*np.log10(np.abs(freq_orig)+1e-10), alpha=0.7, label='Original')
    axes[2, 1].plot(freqs, 20*np.log10(np.abs(freq_proc)+1e-10), alpha=0.7, label='Processed')
    axes[2, 1].set_title('Frequency Response')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude (dB)')
    axes[2, 1].set_xlim(0, 4000)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('noise_reduction_analysis.png', dpi=150)
    plt.show()

def calculate_quality_metrics(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    """
    Calculate objective quality metrics
    """
    # Normalize for comparison
    orig_norm = original / (np.max(np.abs(original)) + 1e-10)
    proc_norm = processed / (np.max(np.abs(processed)) + 1e-10)
    
    # Correlation (voice preservation)
    correlation = np.corrcoef(orig_norm, proc_norm)[0, 1]
    
    # SNR improvement (simplified)
    orig_noise = np.std(original[:16000])  # First second as noise estimate
    proc_noise = np.std(processed[:16000])
    snr_improvement = 20 * np.log10(orig_noise / (proc_noise + 1e-10))
    
    return {
        'correlation': correlation,
        'snr_improvement_db': snr_improvement,
        'rms_reduction': 1 - (np.sqrt(np.mean(processed**2)) / np.sqrt(np.mean(original**2)))
    }
```

### 4. Run Test
```python
# Usage example
if __name__ == "__main__":
    # Test with your WAV file
    input_file = "your_noisy_audio.wav"  # Replace with your file
    output_file = "cleaned_audio.wav"
    
    processed_audio, metrics = test_noise_reduction(input_file, output_file)
    
    # Print quality metrics
    quality = calculate_quality_metrics(audio, processed_audio)
    print(f"\nQuality Metrics:")
    print(f"Voice Correlation: {quality['correlation']:.3f} (target: >0.9)")
    print(f"SNR Improvement: {quality['snr_improvement_db']:.1f} dB")
    print(f"RMS Reduction: {quality['rms_reduction']*100:.1f}%")
```

## Gemini Live API Configuration
```python
# Optimized for reduced interruptions with cleaned audio
gemini_config = {
    "automatic_activity_detection": {
        "disabled": False,
        "start_of_speech_sensitivity": "START_SENSITIVITY_MEDIUM",  # Balanced
        "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",  # Prevent interruptions
        "prefix_padding_ms": 100,
        "silence_duration_ms": 350  # Slightly longer for processed audio
    }
}
```

## Real-time Integration Example
```python
async def handle_plivo_stream(audio_chunk: bytes) -> bytes:
    """
    Real-time processing for Plivo WebSocket
    """
    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
    
    # Process with noise reduction
    cleaned_audio, metrics = reducer.process_chunk(audio_np)
    
    # Log metrics if latency exceeds threshold
    if metrics['latency_ms'] > 15:
        print(f"Warning: High latency {metrics['latency_ms']:.1f}ms")
    
    # Return processed audio as bytes
    return cleaned_audio.tobytes()
```

## Tuning Parameters

### For More Aggressive Noise Reduction
```python
# Increase these values:
prop_decrease = 0.85  # Up to 0.95
noise_gate_threshold = -30  # More aggressive gating
highpass_cutoff = 100  # Remove more low frequencies
```

### For Better Voice Preservation
```python
# Decrease these values:
prop_decrease = 0.6  # Gentler noise reduction
compressor_ratio = 2  # Less compression
gain_db = 1  # Less boost
```

## Performance Optimization
- Process in 10ms chunks (160 samples at 16kHz)
- Use numpy operations (vectorized)
- Pre-allocate buffers for real-time processing
- Monitor CPU usage and adjust parameters if needed

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Robotic voice | Over-aggressive noise reduction | Reduce `prop_decrease` to 0.6 |
| Still noisy | Insufficient reduction | Increase `prop_decrease` to 0.8-0.9 |
| High latency | Complex processing | Disable Compressor or reduce board effects |
| Voice cutting out | Aggressive noise gate | Increase threshold to -40dB |

## Testing Checklist
- [x] Install all dependencies
- [ ] Test with sample WAV file
- [ ] Verify latency <20ms
- [ ] Check voice correlation >0.9
- [ ] Visual inspection of spectrograms
- [ ] Listen to output for quality
- [ ] Test with different noise types
- [ ] Verify Gemini integration works

## Next Steps
1. Run test script with your WAV file
2. Adjust parameters based on results
3. Integrate into WebSocket handler
4. Test with live Plivo stream
5. Monitor Gemini interruption rate