# 🎛️ Noise Reducer Fine-Tuning Guide

## 📋 Overview

This guide helps you optimize the `noise_reducer.py` for your specific audio environment. The system processes audio in real-time chunks and is designed for the Plivo → FastRTC → Gemini Live API pipeline.

## 🚀 Quick Start

### 1. Test Current Performance
```bash
# Test with your audio file
python3 test_noise_reducer.py --input "your_audio.wav"

# Compare different presets
python3 test_noise_reducer.py --compare
```

### 2. Choose Best Preset
- **gentle**: Light noise reduction, excellent voice quality
- **standard**: Balanced approach (recommended starting point)
- **aggressive**: Heavy noise reduction for very noisy environments

### 3. Fine-Tune Parameters
```bash
# Interactive parameter adjustment
python3 test_noise_reducer.py --custom
```

## 🎯 Key Parameters to Adjust

### 1. **Noise Reduction Strength** (`prop_decrease`)
- **Location**: `noise_reducer.py` line 21
- **Range**: 0.5 to 0.95
- **Default**: 0.7 (70% reduction)

**Adjust based on results:**
- **Too much background noise**: Increase to 0.8-0.9
- **Voice sounds robotic**: Decrease to 0.6-0.65
- **Good balance**: Keep at 0.7-0.75

### 2. **High-Pass Filter** (`HighpassFilter`)
- **Location**: `noise_reducer.py` line 27
- **Range**: 60Hz to 150Hz
- **Default**: 80Hz

**Adjust based on noise type:**
- **Traffic/street noise**: Increase to 100-120Hz
- **Office HVAC**: Use 80-90Hz
- **Voice sounds thin**: Decrease to 60-70Hz

### 3. **Noise Gate** (`NoiseGate`)
- **Location**: `noise_reducer.py` lines 30-35
- **Threshold**: -50dB to -25dB
- **Default**: -35dB

**Adjust based on results:**
- **Still noisy in quiet sections**: Decrease to -30dB
- **Voice cuts out**: Increase to -40dB
- **Good balance**: Keep at -35dB

### 4. **Compressor** (`Compressor`)
- **Location**: `noise_reducer.py` lines 37-43
- **Ratio**: 2 to 6
- **Default**: 3

**Adjust based on voice dynamics:**
- **Voice too quiet**: Increase ratio to 4-5
- **Voice sounds compressed**: Decrease ratio to 2-2.5
- **Good dynamics**: Keep at 3

### 5. **Final Gain** (`Gain`)
- **Location**: `noise_reducer.py` line 52
- **Range**: 0dB to 5dB
- **Default**: 2dB

**Adjust based on output level:**
- **Too quiet**: Increase to 3-4dB
- **Too loud/distorted**: Decrease to 1dB
- **Good level**: Keep at 2dB

## 🔧 Parameter Adjustment Process

### Step 1: Test Current Settings
```bash
python3 test_noise_reducer.py --preset standard
```

### Step 2: Identify Issue
- **Voice Quality < 0.85**: Voice is being affected
- **Noise Reduction < 30%**: Not enough noise removal
- **Latency > 20ms**: Too slow for real-time use

### Step 3: Adjust Parameters

#### Problem: Voice Quality Too Low
```python
# In noise_reducer.py, change:
self.prop_decrease = 0.6  # Reduce from 0.7
```

#### Problem: Not Enough Noise Reduction
```python
# In noise_reducer.py, change:
self.prop_decrease = 0.85  # Increase from 0.7
HighpassFilter(cutoff_frequency_hz=100),  # Increase from 80
```

#### Problem: High Latency
```python
# In noise_reducer.py, simplify the board:
self.board = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=80),
    NoiseGate(threshold_db=-35, ratio=4, attack_ms=1, release_ms=50),
    Gain(gain_db=2)
])
```

### Step 4: Test Changes
```bash
python3 test_noise_reducer.py --preset standard
```

### Step 5: Iterate
Repeat steps 3-4 until you achieve target metrics:
- Voice Quality > 0.85
- Noise Reduction > 30%
- Latency < 20ms

## 📊 Understanding Test Results

### Voice Quality (Correlation)
- **> 0.9**: Excellent voice preservation
- **0.85-0.9**: Good voice quality
- **< 0.85**: Voice is being affected, reduce `prop_decrease`

### Noise Reduction (%)
- **> 50%**: Strong noise reduction
- **30-50%**: Good noise reduction
- **< 30%**: Weak noise reduction, increase `prop_decrease`

### Latency (ms)
- **< 15ms**: Excellent for real-time
- **15-20ms**: Good for real-time
- **> 20ms**: May cause delays, simplify processing

## 🎯 Common Scenarios

### Scenario 1: Office Environment
```python
# Moderate noise, consistent background
self.prop_decrease = 0.75
HighpassFilter(cutoff_frequency_hz=90)
NoiseGate(threshold_db=-35)
```

### Scenario 2: Street/Traffic Noise
```python
# High noise, low-frequency dominant
self.prop_decrease = 0.85
HighpassFilter(cutoff_frequency_hz=120)
NoiseGate(threshold_db=-30)
```

### Scenario 3: Quiet Environment
```python
# Minimal noise, preserve voice quality
self.prop_decrease = 0.6
HighpassFilter(cutoff_frequency_hz=70)
NoiseGate(threshold_db=-40)
```

## 🔧 Advanced Tuning

### Custom Presets
Create your own presets by modifying `test_noise_reducer.py`:

```python
# Add to presets dictionary
'my_preset': {
    'prop_decrease': 0.8,
    'time_constant_s': 0.25,
    'highpass_cutoff': 90,
    'noise_gate_threshold': -32,
    'compressor_ratio': 3.5,
    'final_gain': 2.5
}
```

### Batch Testing
Test multiple configurations:
```bash
# Test all presets
python3 test_noise_reducer.py --compare

# Test specific preset
python3 test_noise_reducer.py --preset aggressive
```

## 📈 Production Integration

### Real-Time Processing
```python
from noise_reducer import EnhancedNoiseReducer

# Initialize once
reducer = EnhancedNoiseReducer(sample_rate=16000)

# Process audio chunks
def process_audio_chunk(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    processed, metrics = reducer.process_chunk(audio_np)
    
    # Monitor performance
    if metrics.get('latency_ms', 0) > 20:
        print(f"Warning: High latency {metrics['latency_ms']:.1f}ms")
    
    return processed.tobytes()
```

### Parameter Monitoring
```python
# Track key metrics in production
def monitor_noise_reduction(metrics):
    if 'rms_before' in metrics and 'rms_after' in metrics:
        reduction = 1 - (metrics['rms_after'] / metrics['rms_before'])
        if reduction < 0.2:  # Less than 20% reduction
            print("Warning: Low noise reduction effectiveness")
    
    if metrics.get('latency_ms', 0) > 25:
        print("Warning: Latency too high for real-time processing")
```

## 🚨 Best Practices

### Do's
- ✅ Start with `standard` preset
- ✅ Test with your actual audio samples
- ✅ Monitor latency for real-time use
- ✅ Make small parameter adjustments
- ✅ Test each change individually

### Don'ts
- ❌ Don't set `prop_decrease` > 0.95
- ❌ Don't use high-pass filter > 150Hz
- ❌ Don't ignore latency requirements
- ❌ Don't make multiple changes at once
- ❌ Don't optimize for just one audio sample

## 🎯 Target Metrics

For Plivo → Gemini Live API:
- **Voice Quality**: > 0.85 (excellent > 0.9)
- **Noise Reduction**: > 40% (excellent > 60%)
- **Latency**: < 20ms (excellent < 15ms)

## 📞 Quick Reference

### Test Commands
```bash
# Default test
python3 test_noise_reducer.py

# Compare presets
python3 test_noise_reducer.py --compare

# Custom settings
python3 test_noise_reducer.py --custom

# Specific preset
python3 test_noise_reducer.py --preset aggressive
```

### Key Files
- `noise_reducer.py` - Main implementation (edit this)
- `test_noise_reducer.py` - Testing script
- `output_*.wav` - Test results

The fine-tuning process is iterative. Start with the standard preset, test with your audio, then adjust parameters based on the results until you achieve optimal performance for your specific use case.