"""
Opti-Screen Camera Module - Round-1 Stable Version
Haar Cascade with EMA Smoothing for Jitter Reduction
"""
import cv2
import numpy as np
import os


class Camera:
    """
    Stable camera module for Round-1 demo
    - Haar Cascade face detection only
    - EMA smoothing on bounding box
    - Forehead ROI extraction (top 15-20%)
    - Defensive error handling
    """
    
    def __init__(self, source=None):
        """Initialize camera with video source"""
        self.video = None
        self.dummy_mode = True
        
        # EMA Smoothing variables (The Fix for Haar Jitter)
        self.last_x, self.last_y, self.last_w, self.last_h = 0, 0, 0, 0
        self.alpha = 0.2  # Smoothing factor (Lower = Smoother)
        
        # Video completion flag
        self.video_ended = False
        
        # Initialize Haar Cascade
        try:
            # Try local file first (downloaded)
            cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                # Fallback to OpenCV built-in
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("ERROR: Haar cascade failed to load!")
                self.face_cascade = None
            else:
                print(f"✓ Haar Cascade loaded from: {cascade_path}")
        except Exception as e:
            print(f"ERROR: Haar cascade initialization failed: {e}")
            self.face_cascade = None
        
        # Load video source if provided
        if source and isinstance(source, str):
            try:
                print(f"Opening video file: {source}")
                cap = cv2.VideoCapture(source)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ Successfully opened video: {source}")
                        self.video = cap
                        self.dummy_mode = False
                    else:
                        print(f"Failed to read from video: {source}")
                        cap.release()
                else:
                    print(f"Failed to open video: {source}")
            except Exception as e:
                print(f"Error opening video: {e}")
        else:
            print("No video source. Waiting for upload...")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'video') and self.video is not None:
                self.video.release()
        except:
            pass
    
    def _create_dummy_frame(self):
        """Create placeholder frame"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "WAITING FOR VIDEO", (120, 200), font, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, "Please upload a video file", (100, 250), font, 0.8, (255, 255, 255), 2)
        return frame
    
    def get_frame(self):
        """
        Get frame and extract ROI
        
        Returns:
            (frame_bytes, roi_data)
            roi_data = None if no face detected
            roi_data = (r, g, b) if face detected
        """
        # Dummy mode - waiting for upload
        if self.dummy_mode:
            frame = self._create_dummy_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes(), None
        
        # Read frame
        try:
            success, frame = self.video.read()
            if not success:
                # Video ended - do NOT loop
                self.video_ended = True
                print("[VIDEO] End of video reached")
                return None, None
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None, None
        
        # Extract ROI
        roi_data = self._extract_forehead_roi(frame)
        
        # Encode frame
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes(), roi_data
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None, None
    
    def _extract_forehead_roi(self, frame):
        """
        Extract forehead ROI using Haar Cascade with EMA smoothing
        
        Returns:
            None if no face detected
            (r, g, b) tuple if face detected
        """
        # Safety check
        if self.face_cascade is None:
            cv2.putText(frame, "NO FACE DETECTOR", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None
        
        try:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                # No face detected
                cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return None
            
            # Get largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 1. STABILIZE BOX (Exponential Moving Average)
            if self.last_w == 0:  # First frame
                self.last_x, self.last_y, self.last_w, self.last_h = x, y, w, h
            else:
                self.last_x = int(self.alpha * x + (1 - self.alpha) * self.last_x)
                self.last_y = int(self.alpha * y + (1 - self.alpha) * self.last_y)
                self.last_w = int(self.alpha * w + (1 - self.alpha) * self.last_w)
                self.last_h = int(self.alpha * h + (1 - self.alpha) * self.last_h)
            
            sx, sy, sw, sh = self.last_x, self.last_y, self.last_w, self.last_h
            
            # 2. EXTRACT FOREHEAD ROI (Top 20% of face for better signal)
            fh_x = sx + int(sw * 0.25)  # Center 50% width
            fh_y = sy + int(sh * 0.05)  # Start 5% from top
            fh_w = int(sw * 0.5)        # 50% width
            fh_h = int(sh * 0.2)        # 20% height (larger for better signal)
            
            # Boundary checks
            fh_x = max(0, fh_x)
            fh_y = max(0, fh_y)
            fh_w = min(fh_w, frame.shape[1] - fh_x)
            fh_h = min(fh_h, frame.shape[0] - fh_y)
            
            # Draw visualization
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            cv2.rectangle(frame, (fh_x, fh_y), (fh_x+fh_w, fh_y+fh_h), (255, 0, 0), 3)
            cv2.putText(frame, "FOREHEAD ROI", (fh_x, fh_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Extract signal
            roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
            
            if roi.size == 0:
                return None
            
            # Calculate mean BGR (OpenCV uses BGR, not RGB)
            mean_bgr = np.mean(roi, axis=(0, 1))
            b = float(mean_bgr[0])
            g = float(mean_bgr[1])
            r = float(mean_bgr[2])
            
            # Debug: Print signal values occasionally
            import random
            if random.random() < 0.1:  # 10% of frames
                print(f"[DEBUG] ROI Signal - R:{r:.1f} G:{g:.1f} B:{b:.1f}")
            
            return (r, g, b)
            
        except Exception as e:
            print(f"Error in ROI extraction: {e}")
            return None
