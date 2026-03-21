"""
Opti-Screen Camera Module — MediaPipe Face Mesh Edition
468-landmark face mesh for anatomically precise forehead ROI extraction.
Replaces Haar Cascade for Fitzpatrick I–VI inclusivity.
"""
import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    print("[WARN] mediapipe not installed — falling back to Haar Cascade")


class Camera:
    """
    Camera module with MediaPipe Face Mesh for robust, skin-tone-inclusive
    forehead ROI extraction and landmark-based motion delta.

    Falls back to Haar Cascade if mediapipe is unavailable.
    """

    # MediaPipe forehead landmark indices (top-center cluster)
    _FH_INDICES = [10, 338, 297, 332, 284]

    def __init__(self, source=None):
        """Initialize camera with video source."""
        self.video = None
        self.dummy_mode = True
        self.video_ended = False

        # EMA smoothing (used only in Haar fallback)
        self.last_x = self.last_y = self.last_w = self.last_h = 0
        self.alpha = 0.2
        self.is_moving = False

        # Landmark-based motion tracking
        self._prev_cx = self._prev_cy = None

        # ── Detector init ────────────────────────────────────────────
        self.use_mediapipe = _HAS_MEDIAPIPE
        self.face_mesh = None
        self.face_cascade = None

        if self.use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("✓ MediaPipe Face Mesh initialised")
        else:
            # Haar fallback
            import os
            cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("ERROR: Haar cascade failed to load!")
                self.face_cascade = None
            else:
                print(f"✓ Haar Cascade loaded from: {cascade_path}")

        # ── Open source ──────────────────────────────────────────────
        if source is not None:
            try:
                label = f"video file: {source}" if isinstance(source, str) else f"camera index: {source}"
                print(f"Opening {label}")
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ Successfully opened source: {source}")
                        self.video = cap
                        self.dummy_mode = False
                    else:
                        print(f"Failed to read from source: {source}")
                        cap.release()
                else:
                    print(f"Failed to open source: {source}")
            except Exception as e:
                print(f"Error opening video source: {e}")
        else:
            print("No video source. Waiting for upload…")

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'video') and self.video is not None:
                self.video.release()
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────

    def get_frame(self):
        """
        Get frame and extract ROI.

        Returns
        -------
        (frame_bytes, roi_data, is_moving, motion_delta)
            frame_bytes : bytes | None   — JPEG-encoded frame
            roi_data    : tuple | None   — (r, g, b) forehead means
            is_moving   : bool
            motion_delta: float
        """
        if self.dummy_mode:
            frame = self._create_dummy_frame()
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes(), None, False, 0.0

        try:
            success, frame = self.video.read()
            if not success:
                self.video_ended = True
                print("[VIDEO] End of video reached")
                return None, None, False, 0.0
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None, None, False, 0.0

        # Choose detector
        if self.use_mediapipe:
            roi_data, roi_bgr, motion_delta = self._extract_forehead_mediapipe(frame)
        else:
            roi_data, roi_bgr, motion_delta = self._extract_forehead_haar(frame)

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes(), roi_data, self.is_moving, motion_delta
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None, None, False, 0.0

    # ── MediaPipe extraction ──────────────────────────────────────────

    def _extract_forehead_mediapipe(self, frame):
        """Extract forehead ROI via MediaPipe Face Mesh landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None, None, 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Forehead centroid from anatomically precise landmarks
        fh_x_coords = [int(landmarks[i].x * w) for i in self._FH_INDICES]
        fh_y_coords = [int(landmarks[i].y * h) for i in self._FH_INDICES]
        cx = int(np.mean(fh_x_coords))
        cy = int(np.mean(fh_y_coords))

        # Motion delta via centroid drift
        motion_delta = self._compute_motion_delta(cx, cy)

        # 60×40 px ROI centred on forehead
        roi_w, roi_h = 60, 40
        x1 = max(0, cx - roi_w // 2)
        y1 = max(0, cy - roi_h // 2)
        x2 = min(w, cx + roi_w // 2)
        y2 = min(h, cy + roi_h // 2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, motion_delta

        # Draw visualisation
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "FOREHEAD ROI", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Also draw face bounding box from landmarks for context
        all_x = [int(lm.x * w) for lm in landmarks]
        all_y = [int(lm.y * h) for lm in landmarks]
        face_x1, face_y1 = min(all_x), min(all_y)
        face_x2, face_y2 = max(all_x), max(all_y)
        cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 1)

        mean_bgr = np.mean(roi, axis=(0, 1))
        b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
        return (r, g, b), roi, motion_delta

    # ── Haar fallback ─────────────────────────────────────────────────

    def _extract_forehead_haar(self, frame):
        """Extract forehead ROI using Haar Cascade with EMA smoothing (fallback)."""
        if self.face_cascade is None:
            cv2.putText(frame, "NO FACE DETECTOR", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None, None, 0.0

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return None, None, 0.0

            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            # EMA smoothing
            if self.last_w == 0:
                self.last_x, self.last_y, self.last_w, self.last_h = x, y, w, h
                self.is_moving = False
                motion_delta = 0.0
            else:
                new_x = int(self.alpha * x + (1 - self.alpha) * self.last_x)
                new_y = int(self.alpha * y + (1 - self.alpha) * self.last_y)
                motion_delta = ((new_x - self.last_x) ** 2 + (new_y - self.last_y) ** 2) ** 0.5
                self.is_moving = motion_delta > (self.last_w * 0.03)
                self.last_x, self.last_y = new_x, new_y
                self.last_w = int(self.alpha * w + (1 - self.alpha) * self.last_w)
                self.last_h = int(self.alpha * h + (1 - self.alpha) * self.last_h)

            sx, sy, sw, sh = self.last_x, self.last_y, self.last_w, self.last_h

            fh_x = sx + int(sw * 0.25)
            fh_y = sy + int(sh * 0.05)
            fh_w = int(sw * 0.5)
            fh_h = int(sh * 0.2)
            fh_x = max(0, fh_x)
            fh_y = max(0, fh_y)
            fh_w = min(fh_w, frame.shape[1] - fh_x)
            fh_h = min(fh_h, frame.shape[0] - fh_y)

            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.rectangle(frame, (fh_x, fh_y), (fh_x + fh_w, fh_y + fh_h), (255, 0, 0), 3)
            cv2.putText(frame, "FOREHEAD ROI", (fh_x, fh_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            roi = frame[fh_y:fh_y + fh_h, fh_x:fh_x + fh_w]
            if roi.size == 0:
                return None, None, motion_delta

            mean_bgr = np.mean(roi, axis=(0, 1))
            b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
            return (r, g, b), roi, motion_delta

        except Exception as e:
            print(f"Error in ROI extraction: {e}")
            return None, None, 0.0

    # ── Helpers ───────────────────────────────────────────────────────

    def _compute_motion_delta(self, cx, cy):
        """Compute pixel-wise centroid drift between frames."""
        if self._prev_cx is None:
            self._prev_cx, self._prev_cy = cx, cy
            self.is_moving = False
            return 0.0

        delta = ((cx - self._prev_cx) ** 2 + (cy - self._prev_cy) ** 2) ** 0.5
        self.is_moving = delta > 3.0
        self._prev_cx, self._prev_cy = cx, cy
        return delta

    def _create_dummy_frame(self):
        """Create placeholder frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "WAITING FOR VIDEO", (120, 200), font, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, "Please upload a video file", (100, 250), font, 0.8, (255, 255, 255), 2)
        return frame
