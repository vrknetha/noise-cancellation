# 🎙️ Real-Time Noise Reduction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Overview

Real-time noise reduction system for voice calls, optimized for the Plivo → FastRTC → Gemini Live API pipeline. Processes audio in 10ms chunks with <20ms latency, achieving >85% voice quality preservation while reducing background noise by >40%.

## 🚀 Quick Start

### 1. Setup
```bash
# Clone repository
git clone https://github.com/your-org/noise-reduction-system.git
cd noise-reduction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Test
```bash
# Test with default settings
python test_noise_reducer.py

# Compare different presets
python test_noise_reducer.py --compare

# Interactive parameter tuning
python test_noise_reducer.py --custom
```

### 3. Fine-Tune
```bash
# Edit parameters in noise_reducer.py
# Test changes
python test_noise_reducer.py --preset standard
```

## 📁 Project Structure

```
noise-reduction-system/
├── README.md                          # Project overview
├── SETUP.md                           # Detailed setup guide
├── FINE_TUNING_GUIDE.md               # Parameter tuning instructions
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── pyproject.toml                     # Modern Python packaging
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── noise_reducer.py                   # Main implementation
├── test_noise_reducer.py              # Testing script
├── CLAUDE.md                          # Technical specifications
├── cleaned_audio_ultimate.wav         # Example processed audio
└── WhatsApp Audio 2025-07-11 at 12.30.43 PM.wav  # Test audio
```

## 🎯 Key Features

- **Real-time Processing**: 10ms chunks, <20ms latency
- **Voice Preservation**: >85% correlation with original voice
- **Noise Reduction**: >40% background noise reduction
- **Configurable**: 3 presets + custom parameter tuning
- **Production Ready**: Optimized for Plivo → Gemini pipeline

## 🔧 Core Parameters

Edit these in `noise_reducer.py`:

| Parameter | Line | Purpose | Range | Default |
|-----------|------|---------|-------|---------|
| `prop_decrease` | 21 | Noise reduction strength | 0.5-0.95 | 0.7 |
| `HighpassFilter` | 27 | Remove low-frequency noise | 60-150Hz | 80Hz |
| `NoiseGate` | 30 | Silence quiet sections | -50 to -25dB | -35dB |
| `Compressor` | 37 | Dynamic range control | 2-6 ratio | 3 |
| `Gain` | 52 | Final volume adjustment | 0-5dB | 2dB |

## 🎛️ Available Presets

- **gentle**: Light noise reduction, excellent voice quality
- **standard**: Balanced approach (recommended starting point)
- **aggressive**: Heavy noise reduction for very noisy environments

## 📊 Performance Metrics

### Target Values
- **Voice Quality**: >0.85 (excellent >0.9)
- **Noise Reduction**: >40% (excellent >60%)
- **Latency**: <20ms (excellent <15ms)

### Testing Results
```bash
python test_noise_reducer.py --compare
```

## 🔧 Common Adjustments

### Too Much Background Noise
```python
# In noise_reducer.py, increase noise reduction:
self.prop_decrease = 0.85  # Increase from 0.7
```

### Voice Sounds Robotic
```python
# In noise_reducer.py, reduce processing:
self.prop_decrease = 0.6   # Decrease from 0.7
```

### Street/Traffic Noise
```python
# In noise_reducer.py, filter low frequencies:
HighpassFilter(cutoff_frequency_hz=120)  # Increase from 80
```

## 🚀 Production Integration

```python
from noise_reducer import EnhancedNoiseReducer
import numpy as np

# Initialize once
reducer = EnhancedNoiseReducer(sample_rate=16000)

# Process audio chunks (real-time)
def process_audio_chunk(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    processed, metrics = reducer.process_chunk(audio_np)
    
    # Monitor performance
    if metrics.get('latency_ms', 0) > 20:
        print(f"Warning: High latency {metrics['latency_ms']:.1f}ms")
    
    return processed.tobytes()
```

## 🧪 Testing Commands

```bash
# Test default settings
python test_noise_reducer.py

# Test specific preset
python test_noise_reducer.py --preset aggressive

# Compare all presets
python test_noise_reducer.py --compare

# Custom interactive settings
python test_noise_reducer.py --custom

# Help
python test_noise_reducer.py --help
```

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Complete setup and installation guide
- **[FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md)** - Detailed parameter tuning
- **[CLAUDE.md](CLAUDE.md)** - Technical specifications and requirements

## 🔄 Development Workflow

1. **Setup**: Follow [SETUP.md](SETUP.md) instructions
2. **Edit**: Modify parameters in `noise_reducer.py`
3. **Test**: Run `python test_noise_reducer.py --preset standard`
4. **Iterate**: Repeat until optimal performance
5. **Commit**: `git add . && git commit -m "Optimize noise reduction"`

## 📈 Performance Monitoring

### Key Metrics to Track
```python
# In production code
def monitor_performance(metrics):
    if metrics.get('latency_ms', 0) > 20:
        logger.warning(f"High latency: {metrics['latency_ms']:.1f}ms")
    
    correlation = metrics.get('voice_correlation', 0)
    if correlation < 0.85:
        logger.warning(f"Voice quality degraded: {correlation:.3f}")
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Audio Format Issues**
```bash
# Check file format
python -c "import soundfile as sf; print(sf.info('your_audio.wav'))"
```

**Performance Issues**
```bash
# Test with shorter audio
python test_noise_reducer.py --preset gentle
```

## 📦 Installation

### From Source
```bash
git clone https://github.com/your-org/noise-reduction-system.git
cd noise-reduction-system
pip install -e .
```

### From PyPI (when published)
```bash
pip install noise-reduction-system
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Make changes and test: `python test_noise_reducer.py --compare`
4. Commit changes: `git commit -m "Add feature"`
5. Push to branch: `git push origin feature/improvement`
6. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Use Cases

- **Voice Calls**: Plivo → FastRTC → Gemini Live API
- **Podcasts**: Post-processing audio recordings
- **Live Streaming**: Real-time audio enhancement
- **Call Centers**: Improve call quality

## 🔗 Links

- [GitHub Repository](https://github.com/your-org/noise-reduction-system)
- [Documentation](https://github.com/your-org/noise-reduction-system#readme)
- [Issues](https://github.com/your-org/noise-reduction-system/issues)

---

**Ready to use!** Start with `python test_noise_reducer.py --compare` to find the best preset for your audio environment. 🎉