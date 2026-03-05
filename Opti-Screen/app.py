from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename
from core.camera import Camera
from core.rppg import AdvancedRPPG
import time
import cv2
import numpy as np
import os
import threading

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize components
camera = Camera(source=0)
rppg_engine = AdvancedRPPG(fps=30, window_size=300)

# Global state
current_metrics = {
    'bpm': 0,
    'confidence': 0,
    'status': 'WAITING',
    'snr_db': 0,
    'sqi': 0,
    'mode': 'FACE',  # Always FACE mode for this version
    'classification': 'UNKNOWN',
    'ohi': 0,
    'stability': 0,
    'hrv': 0,
    'stress_index': 0,
    'warnings': [],
    'remark': ''  # Clinical remark (appears when video ends)
}

frame_count = 0
start_time = time.time()
processing_lock = threading.Lock()
is_live_camera = False

def generate_frames():
    """Generator function for video streaming"""
    global frame_count, current_metrics, is_live_camera
    
    while True:
        # Get frame and ROI data from camera
        frame_bytes, roi_data, is_moving = camera.get_frame()
        
        # Check if live camera test duration is completed
        if is_live_camera and (time.time() - start_time > 30):
            print("[APP] Live camera test completed (30s). Stopping...")
            frame_bytes = None  # Force end
            
        # Check if video ended
        if frame_bytes is None:
            # Video ended - generate final summary
            print("[APP] Video ended, generating final summary...")
            
            with processing_lock:
                final_summary = rppg_engine.get_final_summary()
                
                # Update metrics with final stable BPM and remark
                current_metrics.update(final_summary)
                current_metrics['status'] = 'VIDEO_ENDED'
                
                # Update classification based on final BPM
                final_bpm = final_summary['final_bpm']
                if final_bpm < 60:
                    current_metrics['classification'] = 'BRADYCARDIA'
                elif 60 <= final_bpm <= 100:
                    current_metrics['classification'] = 'NORMAL'
                else:
                    current_metrics['classification'] = 'TACHYCARDIA'
            
            print(f"[APP] Final Summary: {final_summary['final_bpm']} BPM - {final_summary['remark']}")
            break  # Stop generating frames
        
        with processing_lock:
            frame_count += 1
            current_time = time.time() - start_time
            
            # Motion warning
            warnings = []
            if is_moving:
                warnings.append("Excessive motion detected. Please stay still.")
            
            # Add frame to rPPG processor
            rppg_engine.add_frame(roi_data, current_time)
            
            # Process rPPG signal
            rppg_results = rppg_engine.process_ppg_signal()
            
            if rppg_results['ready']:
                # Update metrics directly from rPPG results
                bpm = int(rppg_results['bpm'])
                confidence = int(rppg_results['confidence'])
                
                # Determine classification based on BPM
                if bpm < 48:
                    classification = 'BRADYCARDIA'
                elif bpm > 120:
                    classification = 'TACHYCARDIA'
                elif 60 <= bpm <= 100:
                    classification = 'NORMAL'
                else:
                    classification = 'MONITOR'
                
                current_metrics = {
                    'bpm': bpm,
                    'confidence': confidence,
                    'status': rppg_results['status'],
                    'snr_db': rppg_results['snr_db'],
                    'sqi': rppg_results['sqi'],
                    'mode': 'FACE',
                    'classification': classification,
                    'ohi': confidence,  # Use confidence as OHI for simplicity
                    'stability': rppg_results['stability_score'],
                    'stability_indicator': rppg_results['stability_indicator'],
                    'hrv': rppg_results['hrv'],
                    'stress_index': rppg_results['stress_index'],
                    'warnings': warnings,
                    'remark': ''  # Empty during processing, filled at video end
                }
            else:
                current_metrics['status'] = 'CALIBRATING'
                current_metrics['mode'] = 'FACE'
                current_metrics['warnings'] = warnings
        
        # Yield frame for MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.02)  # Cap at ~50 FPS processing

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(current_metrics)

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    """Toggle between FACE and FINGER modes"""
    global current_metrics
    
    with processing_lock:
        # Toggle mode
        if current_metrics['mode'] == 'FACE':
            current_metrics['mode'] = 'FINGER'
        else:
            current_metrics['mode'] = 'FACE'
        
        new_mode = current_metrics['mode']
    
    return jsonify({
        'success': True,
        'mode': new_mode,
        'message': f'Switched to {new_mode} mode'
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_video():
    global camera, rppg_engine, frame_count, start_time, is_live_camera
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Reinitialize components
    with processing_lock:
        camera = Camera(source=filepath)
        rppg_engine = AdvancedRPPG(fps=30, window_size=300)
        frame_count = 0
        start_time = time.time()
        is_live_camera = False
    
    return jsonify({
        'success': True,
        'message': f'Video uploaded: {filename}',
        'filename': filename
    })

@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    global camera, rppg_engine, frame_count, start_time, is_live_camera
    
    with processing_lock:
        camera = Camera(source=None)
        rppg_engine = AdvancedRPPG(fps=30, window_size=300)
        frame_count = 0
        start_time = time.time()
        is_live_camera = False
    
    return jsonify({'success': True, 'message': 'Reset complete'})

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global camera, rppg_engine, frame_count, start_time, is_live_camera
    
    with processing_lock:
        try:
            # Initialize new camera with webcam (index 0)
            camera = Camera(source=0)
            rppg_engine = AdvancedRPPG(fps=30, window_size=300)
            frame_count = 0
            start_time = time.time()
            is_live_camera = True
            
            return jsonify({
                'success': True,
                'message': 'Switched to live webcam'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🫀 Opti-Screen Round-1 Stable Demo")
    print("=" * 60)
    print("✓ Haar Cascade face detection")
    print("✓ POS algorithm for rPPG")
    print("✓ Stable BPM range: 48-120")
    print("=" * 60)
    print("Starting server on http://localhost:5002")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)


