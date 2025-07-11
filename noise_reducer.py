import numpy as np
import noisereduce as nr
from pedalboard import Pedalboard, NoiseGate, Compressor, Gain, LowShelfFilter, HighpassFilter, LowpassFilter, PeakFilter, Limiter
from typing import Tuple, Dict
import time

class EnhancedNoiseReducer:
    """
    Research-backed noise reduction for voice calls
    Optimized for 16kHz PCM audio from Plivo
    Based on optimal configurations from RESEARCH.md
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.frame_count = 0
        self.noise_profile_duration = 0.3  # 300ms for noise profiling
        self.noise_profile_frames = int(self.noise_profile_duration * sample_rate / 160)  # 10ms frames
        
        # Research-backed optimal noise reduction parameters
        self.prop_decrease = 0.6          # Sweet spot between reduction and quality
        self.time_constant_s = 0.15       # Quick adaptation for real-time
        self.freq_mask_smooth_hz = 250    # Voice-optimized smoothing
        
        # Research-backed optimal pedalboard configuration
        self.board = Pedalboard([
            # Stage 1: Remove handling noise (research: 80Hz)
            HighpassFilter(cutoff_frequency_hz=80),
            
            # Stage 2: Aggressive noise gate (research: -45dB)
            NoiseGate(
                threshold_db=-45,      # More aggressive than before
                ratio=10,              # Research: ratio=10
                attack_ms=1,
                release_ms=50
            ),
            
            # Stage 3: Compress dynamics (research: 3:1 ratio)
            Compressor(
                threshold_db=-20,
                ratio=3,               # Research-backed ratio
                attack_ms=5,           # Research: 5ms attack
                release_ms=80          # Research: 80ms release
            ),
            
            # Stage 4: Telephony bandpass (research: 300-3400Hz)
            HighpassFilter(cutoff_frequency_hz=300),     # Telephony band
            LowpassFilter(cutoff_frequency_hz=3400),     # Telephony band
            
            # Stage 5: Presence boost for intelligibility (research: 2kHz)
            PeakFilter(
                cutoff_frequency_hz=2000,
                gain_db=2,
                q=1.5
            ),
            
            # Stage 6: Final limiter to prevent clipping
            Limiter(threshold_db=-3, release_ms=50)
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
        
        # Apply noise reduction with research-backed parameters
        try:
            # Step 1: Statistical noise reduction with optimal parameters
            reduced_audio = nr.reduce_noise(
                y=audio_chunk,
                sr=self.sample_rate,
                y_noise=self.noise_profile,
                prop_decrease=self.prop_decrease,
                stationary=False,              # Non-stationary for dynamic noise
                time_constant_s=self.time_constant_s,
                freq_mask_smooth_hz=self.freq_mask_smooth_hz  # Added from research
            )
            
            # Step 2: Convert to float32 for Pedalboard
            audio_float = reduced_audio.astype(np.float32) / 32768.0
            
            # Step 3: Apply research-backed audio enhancement chain
            processed = self.board(audio_float, self.sample_rate)
            
            # Step 4: Convert back to int16
            processed_int = (processed * 32768).clip(-32768, 32767).astype(np.int16)
            
            # Calculate metrics
            metrics['latency_ms'] = (time.time() - start_time) * 1000
            metrics['rms_before'] = np.sqrt(np.mean(audio_chunk**2) + 1e-10)
            metrics['rms_after'] = np.sqrt(np.mean(processed_int**2) + 1e-10)
            metrics['noise_reduction_db'] = 20 * np.log10(metrics['rms_after'] / max(metrics['rms_before'], 1e-10))
            
            return processed_int, metrics
            
        except Exception as e:
            print(f"Processing error: {e}")
            metrics['error'] = str(e)
            metrics['latency_ms'] = (time.time() - start_time) * 1000
            return audio_chunk, metrics  # Return original on error