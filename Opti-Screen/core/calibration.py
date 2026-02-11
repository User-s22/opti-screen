"""
Calibration module for automatic threshold adjustment per video upload
"""
import cv2
import numpy as np


class VideoCalibrator:
    """Automatically calibrate thresholds based on video characteristics"""
    
    def __init__(self):
        self.baseline_rg_ratio = None
        self.baseline_brightness = None
        self.is_calibrated = False
        
    def calibrate_from_video(self, video_path, calibration_seconds=3):
        """
        Analyze the first few seconds of video to determine baseline values
        
        Args:
            video_path: Path to video file
            calibration_seconds: Number of seconds to analyze (default 3)
            
        Returns:
            dict with calibration parameters
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open video for calibration: {video_path}")
            return self._get_default_calibration()
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default
        
        frames_to_analyze = int(fps * calibration_seconds)
        
        r_values = []
        g_values = []
        b_values = []
        
        frame_count = 0
        while frame_count < frames_to_analyze:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate average RGB for entire frame
            avg_b = np.mean(frame[:, :, 0])
            avg_g = np.mean(frame[:, :, 1])
            avg_r = np.mean(frame[:, :, 2])
            
            r_values.append(avg_r)
            g_values.append(avg_g)
            b_values.append(avg_b)
            
            frame_count += 1
        
        cap.release()
        
        if len(r_values) == 0:
            print("Warning: No frames analyzed, using default calibration")
            return self._get_default_calibration()
        
        # Calculate baseline values
        avg_r = np.mean(r_values)
        avg_g = np.mean(g_values)
        avg_b = np.mean(b_values)
        avg_brightness = (avg_r + avg_g + avg_b) / 3
        
        # Calculate R/G ratio
        rg_ratio = avg_r / avg_g if avg_g > 0 else 2.5
        
        # Store baseline
        self.baseline_rg_ratio = rg_ratio
        self.baseline_brightness = avg_brightness
        self.is_calibrated = True
        
        # Determine thresholds based on baseline
        calibration = self._calculate_thresholds(rg_ratio, avg_brightness)
        
        print(f"\n{'='*60}")
        print(f"VIDEO CALIBRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Analyzed {frame_count} frames ({calibration_seconds} seconds)")
        print(f"Baseline R/G Ratio: {rg_ratio:.3f}")
        print(f"Baseline Brightness: {avg_brightness:.1f}")
        print(f"Detected Mode: {calibration['mode']}")
        print(f"Risk Thresholds:")
        print(f"  - HIGH: > {calibration['high_threshold']:.2f}")
        print(f"  - MODERATE: > {calibration['moderate_threshold']:.2f}")
        print(f"  - LOW: > {calibration['low_threshold']:.2f}")
        print(f"  - VERY_LOW: < {calibration['low_threshold']:.2f}")
        print(f"Optimal Range: {calibration['optimal_min']:.2f} - {calibration['optimal_max']:.2f}")
        print(f"{'='*60}\n")
        
        return calibration
    
    def _calculate_thresholds(self, baseline_rg, brightness):
        """
        Calculate dynamic thresholds based on baseline R/G ratio
        
        Args:
            baseline_rg: Baseline R/G ratio from video
            brightness: Average brightness
            
        Returns:
            dict with threshold values
        """
        # Determine if this is finger or face mode based on characteristics
        is_finger_mode = brightness > 150 or baseline_rg > 1.8
        
        if is_finger_mode:
            # Finger PPG mode - use baseline as center of optimal range
            # Allow ±20% variation as normal
            optimal_min = baseline_rg * 0.85
            optimal_max = baseline_rg * 1.15
            
            # Thresholds relative to baseline
            low_threshold = baseline_rg * 0.7
            moderate_threshold = baseline_rg * 1.2
            high_threshold = baseline_rg * 1.35
            
            mode = "FINGER"
        else:
            # Face mode - lower R/G ratios expected
            optimal_min = max(0.8, baseline_rg * 0.9)
            optimal_max = min(1.5, baseline_rg * 1.1)
            
            low_threshold = baseline_rg * 0.75
            moderate_threshold = baseline_rg * 1.15
            high_threshold = baseline_rg * 1.3
            
            mode = "FACE"
        
        return {
            'mode': mode,
            'baseline_rg': baseline_rg,
            'baseline_brightness': brightness,
            'optimal_min': optimal_min,
            'optimal_max': optimal_max,
            'low_threshold': low_threshold,
            'moderate_threshold': moderate_threshold,
            'high_threshold': high_threshold,
            'is_calibrated': True
        }
    
    def _get_default_calibration(self):
        """Return default calibration if video analysis fails"""
        return {
            'mode': 'FINGER',
            'baseline_rg': 2.5,
            'baseline_brightness': 100,
            'optimal_min': 2.0,
            'optimal_max': 3.0,
            'low_threshold': 1.8,
            'moderate_threshold': 3.0,
            'high_threshold': 3.5,
            'is_calibrated': False
        }
    
    def get_calibration(self):
        """Get current calibration parameters"""
        if not self.is_calibrated:
            return self._get_default_calibration()
        
        return self._calculate_thresholds(self.baseline_rg_ratio, self.baseline_brightness)
