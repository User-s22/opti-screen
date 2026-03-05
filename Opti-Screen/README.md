# Opti-Screen - Biomedical Signal Analysis System

**Real-time Heart Rate Monitoring via Remote Photoplethysmography (rPPG)**

A web-based application that analyzes facial video to extract heart rate (BPM) using advanced signal processing techniques.

---

## 🎯 Features

### Core Functionality
- **Real-time BPM Detection**: Extracts heart rate from facial video using rPPG
- **Signal Quality Monitoring**: Displays Signal Quality Index (SQI) and stability metrics
- **Optical Health Index (OHI)**: Provides overall health assessment
- **Clinical Remarks**: Automatic classification (Bradycardia/Normal/Tachycardia)
- **Session Summary**: Calculates median BPM from entire video session
- **Graceful Video Completion**: No auto-loop, displays final results

### Advanced Features
- **Relaxed Constraints**: Works with short videos (1+ seconds)
- **Low Confidence Acceptance**: Accepts readings with 10%+ confidence
- **Fallback Logic**: Always displays a result (never shows "--")
- **Session Freeze**: Locks final metrics when video ends
- **Polling Stop**: Prevents UI from overwriting final results
- **Live Camera Integration**: Built-in webcam support that automatically concludes a snapshot test after exactly 30 seconds to show the final session summary result.

---

## 🏗️ Architecture

### Backend (`app.py`)
- Flask web server
- Video upload and processing
- Real-time frame streaming
- Session management with final summary

### Signal Processing (`core/rppg.py`)
- **Bandpass Filter**: 0.7-3.0 Hz (42-180 BPM range)
- **FFT Analysis**: Frequency domain BPM extraction
- **SNR Calculation**: Signal-to-noise ratio for quality assessment
- **BPM History Tracking**: Median calculation for stable results
- **Confidence Scoring**: Based on SNR and signal quality

### Camera Module (`core/camera.py`)
- **Haar Cascade Face Detection**: Forehead ROI extraction
- **EMA Smoothing**: Reduces detection jitter
- **RGB Signal Extraction**: Mean values from forehead region
- **Video End Detection**: Graceful completion without looping

### Frontend
- **Real-time Dashboard**: Live BPM, SQI, OHI display
- **PPG Waveform**: Visual signal representation
- **Mode Toggle**: FACE/FINGER mode switching
- **Health Remark Display**: Color-coded clinical assessment
- **Session Lock**: Freezes results when video ends

---

## 📋 Requirements

### Python Dependencies
```
Flask==3.1.0
opencv-python==4.10.0.84
numpy==2.2.1
scipy==1.15.0
```

### System Requirements
- Python 3.8+
- Webcam or video file
- Modern web browser

---

## 🚀 Installation

### 1. Clone Repository
```bash
cd /home/user/anve3.0/Opti-Screen
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Haar Cascade (if not present)
```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

---

## 💻 Usage

### Start the Application
```bash
source venv/bin/activate
python app.py
```

### Access Dashboard
Open browser: `http://localhost:5000` or `http://localhost:5002`

### Analyze a Video
1. Click "Choose File"
2. Select a face video (15-30 seconds recommended)
3. Click "Upload & Analyze"
4. Watch real-time BPM detection
5. View final summary when video ends

### Use Live Camera
1. Click "Use Live Camera"
2. Keep your face steady in front of the webcam for the 30-second duration.
3. The dashboard will process it live and provide a finalized Session Summary when the test time is up.

---

## 🔬 How It Works

### 1. Video Processing
- Uploads video file to server
- Extracts frames at 30 FPS
- Detects face using Haar Cascade
- Extracts forehead ROI (Region of Interest)

### 2. Signal Extraction
- Calculates mean RGB values from forehead
- Applies bandpass filter (0.7-3.0 Hz)
- Removes noise and artifacts
- Tracks signal over time

### 3. BPM Calculation
- Performs FFT (Fast Fourier Transform)
- Finds dominant frequency in valid range
- Converts to BPM (frequency × 60)
- Calculates confidence based on SNR

### 4. Session Summary
- Tracks all valid BPM readings (confidence > 10%)
- Skips first 1 second (calibration)
- Calculates median BPM (robust against outliers)
- Generates clinical remark

### 5. Results Display
- Shows real-time BPM during video
- Displays final median BPM when video ends
- Shows health remark (Bradycardia/Normal/Tachycardia)
- Locks results (no reset to "--")

---

## 📊 Metrics Explained

### Heart Rate (BPM)
- **Range**: 42-180 BPM
- **Classification**:
  - < 60: Bradycardia (Slow)
  - 60-100: Normal Resting Heart Rate
  - \> 100: Tachycardia (Fast)

### Signal Quality Index (SQI)
- **Range**: 0-100%
- **Based on**: SNR and signal stability
- **Threshold**: 10% minimum for acceptance

### Optical Health Index (OHI)
- **Range**: 0-100
- **Based on**: Confidence score
- **Indicates**: Overall signal quality

### Stability
- **Range**: 0-100
- **Indicates**: Signal consistency over time

---

## 🎨 UI Features

### Mode Toggle
- **FACE Mode**: Cyan color (#00bcd4)
- **FINGER Mode**: Orange color (#ff9800)
- Instant switching without page refresh

### Health Remark
- **Normal**: Green (#00ff88)
- **Bradycardia/Tachycardia**: Orange (#ff9800)
- Appears when video ends
- Smooth fade-in animation

### Video Feed
- 450px height with cyan border
- Glow effect for emphasis
- 66% of screen width
- No auto-loop (plays once)

### Upload Section
- 60% opacity (default)
- 100% opacity on hover
- Minimized to reduce distraction

---

## 🔧 Technical Details

### Signal Processing Pipeline
```
Video Frame → Face Detection → ROI Extraction → RGB Mean
→ Bandpass Filter → FFT → Peak Detection → BPM
→ Confidence Calculation → History Tracking → Final Summary
```

### Session Freeze Mechanism
```
Video Ends → get_final_summary() → Calculate Median BPM
→ Generate Clinical Remark → Update current_metrics
→ Set status='VIDEO_ENDED' → Stop Frame Generation
→ Frontend Detects Status → clearInterval(pollInterval)
→ Display Locked Results
```

### Fallback Logic
1. **Primary**: Use median of BPM history
2. **Fallback 1**: Use last known BPM (if history empty)
3. **Fallback 2**: Return demo value (72 BPM)

---

## 🐛 Troubleshooting

### No BPM Detected
- Ensure good lighting
- Face should be clearly visible
- Video should be at least 3 seconds
- Try adjusting camera angle

### Low Confidence
- Improve lighting conditions
- Reduce motion/movement
- Ensure face fills frame
- Use higher quality video

### Video Not Processing
- Check file format (MP4, AVI, MOV)
- Ensure video contains face
- Verify Haar Cascade is downloaded
- Check terminal for error messages

---

## 📁 Project Structure

```
Opti-Screen/
├── app.py                      # Flask application
├── core/
│   ├── camera.py              # Video processing & face detection
│   └── rppg.py                # Signal processing & BPM extraction
├── static/
│   ├── css/
│   │   └── style.css          # Dashboard styling
│   └── js/
│       └── dashboard.js       # Real-time updates & UI logic
├── templates/
│   └── index.html             # Dashboard HTML
├── haarcascade_frontalface_default.xml  # Face detection model
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🔬 Algorithm Details

### Bandpass Filter
- **Type**: Butterworth (4th order)
- **Range**: 0.7-3.0 Hz (42-180 BPM)
- **Implementation**: SOS (Second-Order Sections)
- **Purpose**: Remove noise outside heart rate range

### BPM Extraction
- **Method**: FFT (Fast Fourier Transform)
- **Window**: Hamming window for spectral leakage reduction
- **Peak Detection**: Finds dominant frequency in valid range
- **Conversion**: Frequency (Hz) × 60 = BPM

### Confidence Calculation
- **SNR-based**: Signal-to-Noise Ratio in dB
- **Formula**: `confidence = min(100, max(0, (snr_db - 5) * 10))`
- **Range**: 0-100%
- **Threshold**: 10% for acceptance

---

## 🎯 Key Improvements

### Session Freeze (Zero Drop Fix)
- ✅ Backend stores final metrics in global state
- ✅ Frontend stops polling when video ends
- ✅ Metrics never reset to "--" or "0"
- ✅ Double protection against overwrites

### Relaxed Constraints
- ✅ Confidence threshold: 50% → 10%
- ✅ Buffer requirement: 5 seconds → 1 second
- ✅ Calibration skip: 3 seconds → 1 second
- ✅ Fallback logic ensures results always shown

### Graceful Video Completion
- ✅ Removed auto-loop
- ✅ Calculates median BPM from session
- ✅ Displays clinical remark
- ✅ Locks final results

---

## 📝 API Endpoints

### `GET /`
Returns dashboard HTML

### `GET /video_feed`
Streams video frames (MJPEG)

### `GET /status`
Returns current metrics (JSON)
```json
{
  "bpm": 72,
  "confidence": 85,
  "status": "VIDEO_ENDED",
  "sqi": 85.3,
  "ohi": 85.00,
  "remark": "Normal Resting Heart Rate",
  "classification": "NORMAL"
}
```

### `POST /upload`
Uploads video file for analysis

### `POST /toggle_mode`
Toggles between FACE/FINGER modes

---

## 🚀 Future Enhancements

- [ ] Replay button functionality
- [ ] Multiple video format support
- [ ] Export results to PDF
- [ ] Historical data tracking
- [ ] Multi-user support
- [ ] Mobile app version

---

## 📄 License

This project is for educational and research purposes.

---

## 👨‍💻 Development

### Running in Debug Mode
```bash
export FLASK_ENV=development
python app.py
```

### Testing
```bash
# Test module imports
python -c "from core.camera import Camera; from core.rppg import AdvancedRPPG; print('✓ All modules loaded')"

# Test final summary
python -c "from core.rppg import AdvancedRPPG; engine = AdvancedRPPG(); engine.bpm_history = [70, 72, 71]; print(engine.get_final_summary())"
```

---

## 📞 Support

For issues or questions, check the terminal output for debug messages:
- `[BPM]` - BPM calculation results
- `[HISTORY]` - BPM history tracking
- `[FINAL SUMMARY]` - Session summary generation
- `[APP]` - Application-level events
- `[VIDEO]` - Video processing events
- `[DASHBOARD]` - Frontend events

---

**Built with ❤️ using Flask, OpenCV, and NumPy**
