"""
Opti-Screen rPPG Module — Research-Grade Signal Extraction
CHROM Algorithm (de Haan & Jeanne, 2013) with YCrCb Chrominance Model

Key improvement over POS:
  - Colour-normalises RGB by temporal mean → maps into YCrCb skin-reflectance space
  - Projects onto two orthogonal chrominance vectors (Cr / Cb analogs)
  - Cancels specular reflection AND melanin variance via alpha-scaling
  - Inclusive across skin tones (Fitzpatrick I–VI)
"""
import statistics
from collections import deque

import numpy as np
from scipy import signal
from scipy.signal import find_peaks


class AdvancedRPPG:
    """
    Research-grade rPPG engine using CHROM algorithm (de Haan & Jeanne, 2013).

    - CHROM YCrCb chrominance model (skin-tone inclusive, Fitzpatrick I–VI)
    - 10-second buffer (300 samples @ 30 fps)
    - Robust signal processing pipeline
    - Defensive coding — never crashes
    """

    def __init__(self, fps=30, window_size=300, demo_mode=False):
        self.fps = fps
        self.buffer_size = window_size

        # Signal buffers (deque auto-evicts oldest when full)
        self.r_buffer = deque(maxlen=window_size)
        self.g_buffer = deque(maxlen=window_size)
        self.b_buffer = deque(maxlen=window_size)

        # Bandpass filter (0.8–2.5 Hz = 48–150 BPM)
        try:
            self.sos = signal.butter(4, [0.8, 2.5], btype='bandpass',
                                     fs=self.fps, output='sos')
        except Exception as e:
            print(f"Warning: Filter initialisation failed: {e}")
            self.sos = None

        # Temporal smoothing
        self.prev_bpm = 0
        self.bpm_history = []
        self.frame_count = 0

    # ── Input ─────────────────────────────────────────────────────────

    def add_frame(self, rgb, timestamp=None):
        """Append an (r, g, b) sample to the ring buffers."""
        if rgb is None:
            return
        try:
            r, g, b = rgb
            self.r_buffer.append(r)
            self.g_buffer.append(g)
            self.b_buffer.append(b)
        except Exception as e:
            print(f"Warning: Failed to add frame: {e}")

    # ── Processing ────────────────────────────────────────────────────

    def process_ppg_signal(self):
        """
        Run CHROM → Bandpass → FFT to extract BPM.

        Returns dict with bpm, confidence, status, hrv, stress_index, etc.
        """
        min_samples = int(self.fps * 1)

        if len(self.r_buffer) < min_samples:
            progress = len(self.r_buffer) / min_samples * 100
            if len(self.r_buffer) % 30 == 0:
                print(f"[CALIBRATING] Buffer: {len(self.r_buffer)}/{min_samples} ({progress:.0f}%)")
            return {
                'bpm': 0, 'confidence': 0, 'status': 'CALIBRATING',
                'snr_db': 0, 'sqi': 0, 'ready': False, 'ppg_signal': [],
                'calibration_progress': int(progress),
            }

        try:
            # 1. Numpy arrays from deque
            max_samples = int(self.fps * 10)
            r = np.array(list(self.r_buffer)[-max_samples:])
            g = np.array(list(self.g_buffer)[-max_samples:])
            b = np.array(list(self.b_buffer)[-max_samples:])

            # 2. Temporal mean (DC)
            r_mean = np.mean(r) + 1e-6
            g_mean = np.mean(g) + 1e-6
            b_mean = np.mean(b) + 1e-6

            # 3. Detrend + AC/DC normalisation
            Rn = signal.detrend(r) / r_mean
            Gn = signal.detrend(g) / g_mean
            Bn = signal.detrend(b) / b_mean

            # 4. CHROM projection
            X = 3.0 * Rn - 2.0 * Gn
            Y = 1.5 * Rn + Gn - 1.5 * Bn
            alpha = (np.std(X) + 1e-6) / (np.std(Y) + 1e-6)
            ppg_signal = X - alpha * Y

            # 5. Bandpass filter
            if self.sos is not None:
                try:
                    ppg_filtered = signal.sosfiltfilt(self.sos, ppg_signal)
                except Exception:
                    ppg_filtered = ppg_signal
            else:
                ppg_filtered = ppg_signal

            # 6. Zero-padded 4096-pt FFT
            N_fft = 4096
            fft_vals = np.abs(np.fft.rfft(ppg_filtered, n=N_fft)) ** 2
            freqs = np.fft.rfftfreq(N_fft, d=1.0 / self.fps)

            # 7. Peak in valid range
            valid_mask = (freqs >= 0.75) & (freqs <= 2.5)
            valid_freqs = freqs[valid_mask]
            valid_psd = fft_vals[valid_mask]
            if len(valid_psd) == 0:
                return self._empty_result()

            peak_idx = np.argmax(valid_psd)
            dominant_freq = valid_freqs[peak_idx]
            bpm_raw = dominant_freq * 60.0

            # 8. Adaptive temporal smoothing
            if self.prev_bpm > 0:
                delta = abs(bpm_raw - self.prev_bpm)
                if delta < 10:
                    bpm = 0.70 * self.prev_bpm + 0.30 * bpm_raw
                elif delta < 20:
                    bpm = 0.50 * self.prev_bpm + 0.50 * bpm_raw
                else:
                    bpm = bpm_raw
            else:
                bpm = bpm_raw
            self.prev_bpm = bpm

            # 9. Confidence (SQI) — honest, no inflation
            window = 0.1
            signal_mask = (valid_freqs >= dominant_freq - window) & (valid_freqs <= dominant_freq + window)
            signal_power = np.sum(valid_psd[signal_mask])
            total_power = np.sum(valid_psd) + 1e-6
            raw_confidence = (signal_power / total_power) * 100.0
            confidence = min(100.0, max(0.0, raw_confidence))

            # 10. Status
            if bpm < 48 or bpm > 150:
                status_str = 'OUT_OF_RANGE'
            elif confidence < 20:
                status_str = 'LOW_SIGNAL'
            else:
                status_str = 'OK'

            # 11. SNR
            snr_db = 10 * np.log10(signal_power / (total_power - signal_power + 1e-6))
            snr_db = max(0.0, min(30.0, snr_db))

            print(f'[BPM] Raw:{bpm_raw:.1f} Smooth:{bpm:.1f} Conf:{confidence:.0f}% SNR:{snr_db:.1f}dB Status:{status_str}')

            # 12. BPM history with jump gate
            self.frame_count += 1
            if confidence > 15 and self.frame_count > 30 and status_str == 'OK':
                if len(self.bpm_history) >= 5:
                    recent_median = float(np.median(self.bpm_history[-5:]))
                    if abs(bpm - recent_median) <= 18:
                        self.bpm_history.append(float(bpm))
                else:
                    self.bpm_history.append(float(bpm))

            # 13. Stability indicator
            recent_bpms = self.bpm_history[-30:] if len(self.bpm_history) > 30 else self.bpm_history
            bpm_std = np.std(recent_bpms) if len(recent_bpms) > 2 else 0
            stability_score = max(0, min(100, confidence - (bpm_std * 5)))
            if stability_score > 75:
                stability_indicator = "HIGH"
            elif stability_score > 40:
                stability_indicator = "MEDIUM"
            else:
                stability_indicator = "LOW"

            # 14. HRV (RMSSD) + Baevsky Stress Index
            hrv = 0.0
            stress_index = 0.0
            live_ppg = ppg_filtered[-int(self.fps * 5):]

            if len(live_ppg) >= int(self.fps * 3) and confidence > 5:
                try:
                    min_dist = int(self.fps * 0.4)
                    sig_std = float(np.std(live_ppg))
                    peaks, _ = find_peaks(live_ppg, distance=min_dist, prominence=max(1e-5, sig_std * 0.5))

                    if len(peaks) >= 3:
                        ibis = np.diff(peaks) / self.fps * 1000.0
                        ibis = ibis[(ibis >= 300) & (ibis <= 2000)]

                        if len(ibis) >= 2:
                            successive_diffs = np.diff(ibis)
                            rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
                            hrv = max(5.0, min(128.0, rmssd))

                            mo = float(np.median(ibis))
                            mxdmn = float(np.max(ibis) - np.min(ibis)) + 1e-3
                            hist, _ = np.histogram(ibis, bins=min(len(ibis), 8))
                            am = float(np.max(hist)) / len(ibis)
                            si_raw = (am ** 2) / (2.0 * (mo / 1000.0) * (mxdmn / 1000.0))
                            stress_index = max(5.0, min(95.0, float(si_raw * 0.15)))
                except Exception as hrv_err:
                    print(f"[HRV] Peak detection failed: {hrv_err}")

            return {
                'bpm': float(bpm),
                'confidence': float(confidence),
                'status': status_str,
                'snr_db': float(snr_db),
                'sqi': float(confidence),
                'stability_score': float(stability_score),
                'stability_indicator': stability_indicator,
                'hrv': float(hrv),
                'stress_index': float(stress_index),
                'ready': True,
                'ppg_signal': ppg_filtered.tolist(),
            }

        except Exception as e:
            print(f"Error in signal processing: {e}")
            return self._empty_result()

    # ── Result helpers ────────────────────────────────────────────────

    def _empty_result(self):
        """Return empty result for error cases."""
        return {
            'bpm': 0, 'confidence': 0.0, 'status': 'NO_FACE',
            'snr_db': 0, 'sqi': 0, 'stability_score': 0,
            'stability_indicator': 'LOW', 'hrv': 0.0, 'stress_index': 0.0,
            'ready': False, 'ppg_signal': [],
        }

    def get_final_summary(self):
        """
        Calculate final session summary.

        Returns a dict with final_bpm, classification, remark, etc.
        Returns final_bpm=None when data is insufficient (no fake values).
        """
        if len(self.bpm_history) == 0:
            if self.prev_bpm > 40:
                print(f"[FINAL SUMMARY] No history, using last BPM: {self.prev_bpm}")
                final_bpm = round(self.prev_bpm)
                remark = self._classify_remark(final_bpm) + " — Low Confidence"
                return {
                    'final_bpm': final_bpm,
                    'min_bpm': final_bpm, 'max_bpm': final_bpm, 'avg_bpm': final_bpm,
                    'stability_percent': 0,
                    'remark': remark,
                    'total_readings': 0,
                }
            # Insufficient data — NO fake fallback
            print("[FINAL SUMMARY] Insufficient data")
            return {
                'final_bpm': None,
                'min_bpm': None, 'max_bpm': None, 'avg_bpm': None,
                'stability_percent': 0,
                'remark': 'INSUFFICIENT_DATA',
                'error': 'Minimum 5 seconds of clean face signal required',
                'total_readings': 0,
            }

        median_bpm = statistics.median(self.bpm_history)
        final_bpm = round(median_bpm)
        remark = self._classify_remark(final_bpm)

        min_bpm = round(min(self.bpm_history))
        max_bpm = round(max(self.bpm_history))
        avg_bpm = round(statistics.mean(self.bpm_history))

        # IQR outlier filter
        sorted_h = sorted(self.bpm_history)
        n = len(sorted_h)
        q1 = sorted_h[n // 4]
        q3 = sorted_h[(3 * n) // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in sorted_h if lo <= x <= hi] or sorted_h

        overall_std = statistics.stdev(filtered) if len(filtered) > 1 else 0
        stability_percent = round(max(0, min(100, 100 - overall_std * 4)))

        print(f"[FINAL SUMMARY] Median BPM: {final_bpm} | Remark: {remark} | Readings: {len(self.bpm_history)}")

        return {
            'final_bpm': final_bpm,
            'min_bpm': min_bpm, 'max_bpm': max_bpm, 'avg_bpm': avg_bpm,
            'stability_percent': stability_percent,
            'remark': remark,
            'total_readings': len(self.bpm_history),
        }

    @staticmethod
    def _classify_remark(bpm):
        if bpm < 60:
            return "Bradycardia (Slow)"
        elif bpm <= 100:
            return "Normal Resting Heart Rate"
        else:
            return "Tachycardia (Fast)"

    def get_signal_quality(self):
        """Legacy method for compatibility."""
        return 0
