"""
Opti-Screen rPPG Module - Round-1 Stable Version
POS Algorithm with Robust Signal Processing
"""
import numpy as np
from scipy import signal


class AdvancedRPPG:
    """
    Stable rPPG engine for Round-1 demo
    - POS algorithm only
    - 10-second buffer (300 samples @ 30fps)
    - Robust signal processing pipeline
    - Defensive coding - never crashes
    """
    
    def __init__(self, fps=30, window_size=300, demo_mode=False):
        """
        Initialize rPPG processor
        
        Args:
            fps: Frame rate (default 30)
            window_size: Buffer size in frames (default 300 = 10 seconds)
            demo_mode: Unused, kept for compatibility
        """
        self.fps = fps
        self.buffer_size = window_size
        
        # Signal buffers
        self.r_buffer = []
        self.g_buffer = []
        self.b_buffer = []
        
        # Bandpass filter (0.7-3.0 Hz = 42-180 BPM) - Wider range for better detection
        # Using SOS (second-order sections) for numerical stability
        try:
            self.sos = signal.butter(4, [0.7, 3.0], btype='bandpass', 
                                    fs=self.fps, output='sos')
        except Exception as e:
            print(f"Warning: Filter initialization failed: {e}")
            self.sos = None
        
        # Smoothing for stable BPM
        self.prev_bpm = 0
        self.bpm_history = []  # Track all valid BPM readings for final summary
        self.frame_count = 0  # Track frames for calibration skip
    
    def add_frame(self, rgb, timestamp=None):
        """
        Add RGB sample to buffer
        
        Args:
            rgb: Tuple of (r, g, b) or None
            timestamp: Unused, kept for compatibility
        """
        if rgb is None:
            return
        
        try:
            r, g, b = rgb
            
            # Add to buffers
            self.r_buffer.append(r)
            self.g_buffer.append(g)
            self.b_buffer.append(b)
            
            # Keep buffer fixed size
            if len(self.r_buffer) > self.buffer_size:
                self.r_buffer.pop(0)
                self.g_buffer.pop(0)
                self.b_buffer.pop(0)
        except Exception as e:
            print(f"Warning: Failed to add frame: {e}")
    
    def process_ppg_signal(self):
        """
        Process buffered signal and extract BPM
        
        Returns:
            dict with keys:
                - bpm: Heart rate in BPM (0 if insufficient data)
                - confidence: Confidence percentage (0-100)
                - status: "OK", "LOW_SIGNAL", or "NO_FACE"
                - snr_db: Signal-to-noise ratio
                - sqi: Signal quality index
                - ready: True if valid result
        """
        # Check if we have enough samples (relaxed to 1 second for faster results)
        min_samples = int(self.fps * 1)  # Changed from 5 to 1 second
        
        if len(self.r_buffer) < min_samples:
            # Debug: Show calibration progress
            progress = len(self.r_buffer) / min_samples * 100
            if len(self.r_buffer) % 30 == 0:  # Every second
                print(f"[CALIBRATING] Buffer: {len(self.r_buffer)}/{min_samples} ({progress:.0f}%)")
            
            return {
                'bpm': 0,
                'confidence': 0,
                'status': 'CALIBRATING',
                'snr_db': 0,
                'sqi': 0,
                'ready': False,
                'ppg_signal': []
            }
        
        try:
            # 1. Convert to numpy arrays (use last 10 seconds max)
            max_samples = int(self.fps * 10)
            r = np.array(self.r_buffer[-max_samples:])
            g = np.array(self.g_buffer[-max_samples:])
            b = np.array(self.b_buffer[-max_samples:])
            
            # 2. DETRENDING (Remove light drift)
            r = signal.detrend(r)
            g = signal.detrend(g)
            b = signal.detrend(b)
            
            # 3. NORMALIZATION (Standardization - mean=0, std=1)
            r_mean = np.mean(r)
            g_mean = np.mean(g)
            b_mean = np.mean(b)
            r_std = np.std(r) + 1e-6
            g_std = np.std(g) + 1e-6
            b_std = np.std(b) + 1e-6
            
            rn = (r - r_mean) / r_std
            gn = (g - g_mean) / g_std
            bn = (b - b_mean) / b_std
            
            # 4. POS ALGORITHM (Plane-Orthogonal-to-Skin)
            # Build chrominance signals
            X = rn - gn
            Y = rn + gn - 2*bn
            
            # Alpha tuning (with zero-division protection)
            X_std = np.std(X) + 1e-6
            Y_std = np.std(Y) + 1e-6
            alpha = X_std / Y_std
            
            # Fuse signals
            ppg_signal = X - alpha * Y
            
            # 5. BANDPASS FILTER (0.75Hz - 2.5Hz)
            if self.sos is None:
                # Fallback if filter failed to initialize
                ppg_filtered = ppg_signal
            else:
                try:
                    ppg_filtered = signal.sosfiltfilt(self.sos, ppg_signal)
                except Exception as e:
                    print(f"Warning: Filtering failed: {e}")
                    ppg_filtered = ppg_signal
            
            # 6. WELCH'S METHOD (Frequency Analysis)
            try:
                nperseg = min(len(ppg_filtered), 256)
                freqs, psd = signal.welch(ppg_filtered, fs=self.fps, nperseg=nperseg)
            except Exception as e:
                print(f"Warning: Welch failed: {e}")
                return self._empty_result()
            
            # 7. Find Peak in Valid Range (0.7 - 3.0 Hz = 42 - 180 BPM)
            valid_idx = np.where((freqs >= 0.7) & (freqs <= 3.0))
            valid_freqs = freqs[valid_idx]
            valid_psd = psd[valid_idx]
            
            if len(valid_psd) == 0:
                return self._empty_result()
            
            # Find dominant frequency
            peak_idx = np.argmax(valid_psd)
            dominant_freq = valid_freqs[peak_idx]
            
            # Convert to BPM
            bpm_raw = dominant_freq * 60.0
            
            # 8. TEMPORAL SMOOTHING (for stability)
            if self.prev_bpm > 0 and abs(bpm_raw - self.prev_bpm) < 20:
                # Smooth with previous reading (80% new, 20% old)
                bpm = 0.8 * bpm_raw + 0.2 * self.prev_bpm
            else:
                bpm = bpm_raw
            
            self.prev_bpm = bpm
            
            # 9. Calculate Confidence
            # Confidence = peak_power / total_band_power
            peak_power = np.max(valid_psd)
            total_power = np.sum(valid_psd) + 1e-6
            confidence = (peak_power / total_power) * 100.0
            confidence = min(100.0, max(0.0, confidence))
            
            # 10. Determine Status
            if bpm < 48 or bpm > 120:
                status = "OUT_OF_RANGE"
            elif confidence < 25:
                status = "LOW_SIGNAL"
            else:
                status = "OK"
            
            # 11. Calculate SNR (simple approximation)
            snr_db = 10 * np.log10(peak_power / (total_power - peak_power + 1e-6))
            snr_db = max(0, min(30, snr_db))  # Clamp to reasonable range
            
            # Debug output
            print(f"[BPM] {bpm:.1f} BPM | Confidence: {confidence:.1f}% | SNR: {snr_db:.1f} dB | Status: {status}")
            
            # Track BPM history for final summary (relaxed to 10% confidence)
            self.frame_count += 1
            if confidence > 10 and self.frame_count > 30:  # Skip first 1 second (30 frames @ 30fps)
                self.bpm_history.append(bpm)
                print(f"[HISTORY] Added BPM {bpm:.1f} to history (size: {len(self.bpm_history)})")
            
            return {
                'bpm': float(bpm),
                'confidence': float(confidence),
                'status': status,
                'snr_db': float(snr_db),
                'sqi': float(confidence),  # Use confidence as SQI for simplicity
                'ready': True,
                'ppg_signal': ppg_filtered.tolist()
            }
            
        except Exception as e:
            print(f"Error in signal processing: {e}")
            return self._empty_result()
    
    def _empty_result(self):
        """Return empty result for error cases"""
        return {
            'bpm': 0,
            'confidence': 0.0,
            'status': 'NO_FACE',
            'snr_db': 0,
            'sqi': 0,
            'ready': False,
            'ppg_signal': []
        }
    
    def get_final_summary(self):
        """
        Calculate final session summary with median BPM and clinical remark
        
        Returns:
            dict with keys:
                - final_bpm: Median BPM from session (excluding calibration)
                - remark: Clinical remark based on BPM range
                - total_readings: Number of valid readings used
        """
        if len(self.bpm_history) == 0:
            # Fallback 1: Use last known BPM if available
            if self.prev_bpm > 40:
                print(f"[FINAL SUMMARY] No history, using last BPM: {self.prev_bpm}")
                final_bpm = round(self.prev_bpm)
                
                # Generate remark based on last BPM
                if final_bpm < 60:
                    remark = "Bradycardia (Slow) - Low Confidence"
                elif 60 <= final_bpm <= 100:
                    remark = "Normal Resting Heart Rate - Low Confidence"
                else:
                    remark = "Tachycardia (Fast) - Low Confidence"
                
                return {
                    'final_bpm': final_bpm,
                    'remark': remark,
                    'total_readings': 0
                }
            
            # Fallback 2: Return demo value
            print("[FINAL SUMMARY] No data available, returning demo value")
            return {
                'final_bpm': 72,
                'remark': 'Demo Value - Insufficient Data',
                'total_readings': 0
            }
        
        # Calculate median BPM (more robust than mean)
        import statistics
        median_bpm = statistics.median(self.bpm_history)
        final_bpm = round(median_bpm)
        
        # Clinical remark based on medical standards
        if final_bpm < 60:
            remark = "Bradycardia (Slow)"
        elif 60 <= final_bpm <= 100:
            remark = "Normal Resting Heart Rate"
        else:  # > 100
            remark = "Tachycardia (Fast)"
        
        print(f"[FINAL SUMMARY] Median BPM: {final_bpm} | Remark: {remark} | Readings: {len(self.bpm_history)}")
        
        return {
            'final_bpm': final_bpm,
            'remark': remark,
            'total_readings': len(self.bpm_history)
        }
    
    def get_signal_quality(self):
        """Legacy method for compatibility"""
        return 0
