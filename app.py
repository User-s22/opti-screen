from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from core.camera import Camera
from core.rppg import AdvancedRPPG
import statistics as stats
import time, cv2, numpy as np, os, threading

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Global components ────────────────────────────────────────────────# Initialize components — camera starts in dummy mode; activated on user action
camera       = Camera(source=None)   # No hardware opened until user chooses mode
rppg_engine  = AdvancedRPPG(fps=30, window_size=300)

processing_lock = threading.Lock()
stop_event      = threading.Event()
frame_count     = 0
start_time      = time.time()
is_live_camera  = False

# Tracks all valid BPM readings this session for summary stats
bpm_history: list[float] = []

# Persist last known good HRV and stress values
# (peak detection may fail on a given frame — we never want to show 0 in the summary)
last_valid_hrv:    float = 0.0
last_valid_stress: float = 0.0

current_metrics = {
    'bpm': 0, 'confidence': 0, 'status': 'WAITING',
    'snr_db': 0, 'sqi': 0, 'classification': 'UNKNOWN',
    'ohi': 0, 'stability': 0, 'stability_indicator': '--',
    'hrv': 0, 'stress_index': 0, 'warnings': [],
    'remark': '', 'estimated_lux': 0, 'motion_delta': 0,
    'is_live': False, 'calibration_done': False,
    'ppg_signal': [], 'calibration_progress': 0,
}


def _classify(bpm):
    """
    Clinical heart rate classification (resting, adult):
      < 60        → BRADYCARDIA   (slow)
      60 – 90     → NORMAL        (user-defined normal ceiling)
      91 – 100    → MONITOR       (elevated / borderline)
      > 100       → TACHYCARDIA   (fast)
    """
    if not isinstance(bpm, (int, float)):
        return '--'
    if bpm < 60:
        return 'BRADYCARDIA'
    if bpm <= 90:
        return 'NORMAL'
    if bpm <= 100:
        return 'MONITOR'       # elevated but not tachycardia
    return 'TACHYCARDIA'


def generate_frames():
    """MJPEG generator. Runs for the lifetime of the /video_feed request."""
    global frame_count, current_metrics, bpm_history, last_valid_hrv, last_valid_stress

    calibration_done = False

    while not stop_event.is_set():
        frame_bytes, roi_data, is_moving, motion_delta = camera.get_frame()

        # ── Video file ended ─────────────────────────────────────────
        if frame_bytes is None:
            print("[APP] Video ended — computing final summary")
            with processing_lock:
                final = rppg_engine.get_final_summary()
                current_metrics.update(final)
                current_metrics['status']       = 'VIDEO_ENDED'
                current_metrics['classification'] = _classify(final.get('final_bpm'))
                current_metrics['calibration_done'] = calibration_done
            break

        with processing_lock:
            frame_count += 1
            elapsed = time.time() - start_time

            # Perceptual luminance (ITU-R BT.601) — accurate pseudo-lux
            if roi_data is not None:
                r_ch, g_ch, b_ch = roi_data
                lux = int(0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch)
            else:
                lux = 0
            current_metrics['estimated_lux'] = lux
            current_metrics['motion_delta']  = int(motion_delta)

            # Warnings list
            warnings = []
            if is_moving or motion_delta > 15.0:
                warnings.append("Excessive motion — please stay still")
            if roi_data is not None and (lux < 50 or lux > 210):
                warnings.append("Poor lighting — move to better light")

            rppg_engine.add_frame(roi_data, elapsed)
            results = rppg_engine.process_ppg_signal()

            if results['ready']:
                calibration_done = True
                bpm = results.get('bpm', 0)
                if isinstance(bpm, (int, float)) and bpm > 0:
                    bpm_history.append(float(bpm))

                # Persist last known-good HRV and stress
                hrv_now    = results.get('hrv', 0) or 0
                stress_now = results.get('stress_index', 0) or 0
                if hrv_now    > 0: last_valid_hrv    = hrv_now
                if stress_now > 0: last_valid_stress = stress_now

                current_metrics = {
                    'bpm':                  int(bpm) if isinstance(bpm, (int, float)) else 0,
                    'confidence':           int(results.get('confidence', 0)),
                    'status':               results.get('status', 'OK'),
                    'snr_db':               results.get('snr_db', 0),
                    'sqi':                  results.get('sqi', 0),
                    'classification':       _classify(bpm),
                    'ohi':                  results.get('confidence', 0),
                    'stability':            results.get('stability_score', 0),
                    'stability_indicator':  results.get('stability_indicator', '--'),
                    'hrv':                  last_valid_hrv,
                    'stress_index':         last_valid_stress,
                    'warnings':             warnings,
                    'remark':               results.get('remark', ''),
                    'estimated_lux':        lux,
                    'motion_delta':         int(motion_delta),
                    'is_live':              is_live_camera,
                    'calibration_done':     True,
                    'face_detected':        roi_data is not None,
                    'ppg_signal':           results.get('ppg_signal', [])[-150:],
                    'calibration_progress': 100,
                }
            else:
                current_metrics['status']           = 'CALIBRATING'
                current_metrics['warnings']         = warnings
                current_metrics['is_live']          = is_live_camera
                current_metrics['calibration_done'] = False
                current_metrics['estimated_lux']    = lux
                current_metrics['motion_delta']     = int(motion_delta)
                current_metrics['face_detected']    = roi_data is not None
                current_metrics['calibration_progress'] = results.get('calibration_progress', 0)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.02)


# ── Routes ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with processing_lock:
        payload = dict(current_metrics)
    return jsonify(payload)

@app.route('/session_summary')
def session_summary():
    """Returns final session statistics built from bpm_history."""
    with processing_lock:
        history = list(bpm_history)
        metrics = dict(current_metrics)

    if len(history) >= 2:
        sorted_h = sorted(history)
        n = len(sorted_h)
        q1 = sorted_h[n // 4]
        q3 = sorted_h[(3 * n) // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clean = [x for x in sorted_h if lo <= x <= hi] or sorted_h
        avg_bpm = round(stats.mean(clean), 1)
        min_bpm = round(min(clean), 1)
        max_bpm = round(max(clean), 1)
        std_bpm = stats.stdev(clean) if len(clean) > 1 else 0
        stability_pct = round(max(0, min(100, 100 - std_bpm * 4)))
    elif len(history) == 1:
        avg_bpm = min_bpm = max_bpm = round(history[0], 1)
        stability_pct = 0
    else:
        # INSUFFICIENT_DATA — no fake values
        return jsonify({
            'avg_bpm': None, 'min_bpm': None, 'max_bpm': None,
            'stability_pct': 0,
            'hrv': round(metrics.get('hrv', 0), 1),
            'stress_index': round(metrics.get('stress_index', 0), 1),
            'classification': '--',
            'remark': 'INSUFFICIENT_DATA',
            'sample_count': 0,
        })

    return jsonify({
        'avg_bpm':        avg_bpm,
        'min_bpm':        min_bpm,
        'max_bpm':        max_bpm,
        'stability_pct':  stability_pct,
        'hrv':            round(metrics.get('hrv', 0), 1),
        'stress_index':   round(metrics.get('stress_index', 0), 1),
        'classification': _classify(avg_bpm),
        'remark':         metrics.get('remark', ''),
        'sample_count':   len(history),
    })


def _reset(source, live):
    global camera, rppg_engine, frame_count, start_time, is_live_camera, bpm_history, last_valid_hrv, last_valid_stress
    stop_event.set()           # Signal the running generator to stop
    time.sleep(0.15)           # Give it a moment to exit
    stop_event.clear()
    with processing_lock:
        if camera.video is not None:
            camera.video.release()   # Explicit release before reassignment
        camera         = Camera(source=source)
        rppg_engine    = AdvancedRPPG(fps=30, window_size=300)
        frame_count    = 0
        start_time     = time.time()
        is_live_camera = live
        bpm_history    = []
        last_valid_hrv    = 0.0
        last_valid_stress = 0.0
        current_metrics.update({
            'bpm': 0, 'status': 'WAITING', 'classification': 'UNKNOWN',
            'hrv': 0, 'stress_index': 0, 'warnings': [],
            'estimated_lux': 0, 'motion_delta': 0,
            'is_live': live, 'calibration_done': False,
        })


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    filename  = secure_filename(file.filename)
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    _reset(filepath, live=False)
    return jsonify({'success': True, 'message': f'Video uploaded: {filename}'})

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    try:
        _reset(0, live=True)
        return jsonify({'success': True, 'message': 'Live camera started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    _reset(None, live=False)
    return jsonify({'success': True, 'message': 'Reset complete'})


if __name__ == '__main__':
    print("=" * 60)
    print("🫀 Opti-Screen - Research-Grade rPPG")
    print("=" * 60)
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=5002, threaded=True, use_reloader=False)
