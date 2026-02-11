"""
Production-Grade Signal Processing for Accurate Vital Signs
Implements research-proven algorithms for PPG analysis
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from collections import deque


class SignalEngine:
    """Enhanced signal processing engine with FFT-based heart rate detection"""
    
    def __init__(self, buffer_size=300, fps=30):
        """
        Initialize signal engine
        
        Args:
            buffer_size: Number of frames to buffer (default 300 = 10s at 30fps)
            fps: Frames per second
        """
        self.buffer_size = buffer_size
        self.fps = fps
        
        # Circular buffers for RGB channels
        self.red_buffer = deque(maxlen=buffer_size)
        self.green_buffer = deque(maxlen=buffer_size)
        self.blue_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # Design Butterworth bandpass filter
        # Frequency range: 0.75-3.5 Hz (45-210 BPM)
        nyquist = 0.5 * fps
        low = 0.75 / nyquist
        high = 3.5 / nyquist
        self.b, self.a = scipy_signal.butter(5, [low, high], btype='band')
    
    def process_sample(self, r, g, b, timestamp=None):
        """
        Add new sample and return filtered signal
        
        Args:
            r, g, b: RGB channel values
            timestamp: Optional timestamp
            
        Returns:
            dict with filtered signals and FFT-based BPM
        """
        # Add to buffers
        self.red_buffer.append(r)
        self.green_buffer.append(g)
        self.blue_buffer.append(b)
        
        if timestamp is None:
            if len(self.timestamps) == 0:
                timestamp = 0
            else:
                timestamp = self.timestamps[-1] + (1.0 / self.fps)
        self.timestamps.append(timestamp)
        
        # Need minimum samples
        if len(self.green_buffer) < 60:  # At least 2 seconds
            return {
                'filtered_green': [],
                'filtered_red': [],
                'raw_green': list(self.green_buffer),
                'raw_red': list(self.red_buffer),
                'fft_bpm': 0,
                'ready': False
            }
        
        # Convert to numpy arrays
        green_signal = np.array(self.green_buffer)
        red_signal = np.array(self.red_buffer)
        
        # Detrend to remove slow baseline drift
        green_detrended = scipy_signal.detrend(green_signal)
        red_detrended = scipy_signal.detrend(red_signal)
        
        # Apply Butterworth bandpass filter
        filtered_green = scipy_signal.filtfilt(self.b, self.a, green_detrended)
        filtered_red = scipy_signal.filtfilt(self.b, self.a, red_detrended)
        
        # FFT-based heart rate detection (more accurate than peak detection)
        fft_bpm = self._calculate_fft_bpm(filtered_green)
        
        return {
            'filtered_green': filtered_green,
            'filtered_red': filtered_red,
            'raw_green': green_signal,
            'raw_red': red_signal,
            'fft_bpm': fft_bpm,
            'ready': True
        }
    
    def _calculate_fft_bpm(self, signal_data):
        """
        Calculate BPM using FFT (frequency domain analysis)
        More accurate and robust than time-domain peak detection
        
        Args:
            signal_data: Filtered signal
            
        Returns:
            BPM value
        """
        if len(signal_data) < 60:
            return 0
        
        # Apply Hamming window to reduce spectral leakage
        windowed = signal_data * np.hamming(len(signal_data))
        
        # Compute FFT
        N = len(windowed)
        yf = fft(windowed)
        xf = fftfreq(N, 1/self.fps)
        
        # Only positive frequencies
        positive_freqs = xf[:N//2]
        positive_fft = np.abs(yf[:N//2])
        
        # Focus on physiological range: 0.75-3.5 Hz (45-210 BPM)
        valid_idx = (positive_freqs >= 0.75) & (positive_freqs <= 3.5)
        valid_freqs = positive_freqs[valid_idx]
        valid_fft = positive_fft[valid_idx]
        
        if len(valid_fft) == 0:
            return 0
        
        # Find dominant frequency
        peak_idx = np.argmax(valid_fft)
        dominant_freq = valid_freqs[peak_idx]
        
        # Convert to BPM
        bpm = dominant_freq * 60
        
        # Additional validation: check if peak is significant
        mean_power = np.mean(valid_fft)
        peak_power = valid_fft[peak_idx]
        
        # Peak should be at least 2x the mean power
        if peak_power < 2 * mean_power:
            return 0
        
        return bpm
    
    def get_signal_quality(self, signal_data):
        """
        Calculate signal quality metrics
        
        Args:
            signal_data: Filtered signal
            
        Returns:
            Quality score (0-100)
        """
        if len(signal_data) < 30:
            return 0
        
        # Calculate SNR (Signal-to-Noise Ratio)
        signal_power = np.var(signal_data)
        
        # Estimate noise from high-frequency components
        diff = np.diff(signal_data)
        noise_power = np.var(diff)
        
        if noise_power == 0:
            snr = 100
        else:
            snr = 10 * np.log10(signal_power / noise_power)
        
        # Convert SNR to 0-100 scale
        # SNR > 20 dB is excellent, < 5 dB is poor
        quality = np.clip((snr - 5) / 15 * 100, 0, 100)
        
        return quality
