from collections import deque
import numpy as np

class BPMSmoother:
    """
    Stabilizes BPM accuracy using outlier rejection and exponential moving average.
    """
    
    def __init__(self, history_size=10, max_change=12.0):
        """
        Args:
            history_size: Number of readings to keep for smoothing
            max_change: Maximum allowed jump in BPM between frames (outlier rejection)
        """
        self.history = deque(maxlen=history_size)
        self.max_change = max_change
        self.current_bpm = 0.0
        
    def update(self, new_bpm):
        """
        Update with new BPM reading and return smoothed value
        """
        if new_bpm <= 0:
            return self.current_bpm
            
        # Initialize if empty
        if self.current_bpm == 0:
            self.current_bpm = new_bpm
            self.history.append(new_bpm)
            return new_bpm
            
        # Outlier rejection
        # If change is too sudden, ignore it (unless it persists)
        if abs(new_bpm - self.current_bpm) > self.max_change:
            # Check if this "outlier" is actually a trend (persists in recent history)
            # If the last 3 readings were also "outliers" close to this one, accept it
            if len(self.history) >= 3:
                recent_avg = np.mean(list(self.history)[-3:])
                if abs(new_bpm - recent_avg) < self.max_change:
                    # It's a valid state change, accept it
                    pass
                else:
                    # truly an outlier, ignore
                    return self.current_bpm
            else:
                 return self.current_bpm
        
        # Add to history
        self.history.append(new_bpm)
        
        # Calculate Weighted Moving Average
        # Give more weight to recent readings
        readings = list(self.history)
        weights = np.linspace(0.5, 1.0, len(readings))
        weights /= np.sum(weights)
        
        smoothed_bpm = np.average(readings, weights=weights)
        self.current_bpm = smoothed_bpm
        
        return smoothed_bpm
