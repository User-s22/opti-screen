<div align="center">
  <img src="https://img.shields.io/badge/Opti--Screen-v4.0-00ffcc?style=for-the-badge&logo=medicare" alt="Version">
  <img src="https://img.shields.io/badge/Platform-Ubuntu%20%7C%20Windows-blue?style=for-the-badge&logo=linux" alt="Platform">
  <img src="https://img.shields.io/badge/Algorithm-CHROM%20rPPG-ff6600?style=for-the-badge" alt="Algorithm">
  <img src="https://img.shields.io/badge/Face%20Detection-MediaPipe%20Mesh-8e44ad?style=for-the-badge" alt="Detector">
</div>

<br />

<div align="center">
  <h1>🔬 Opti-Screen: Research-Grade Contactless Vitals via CHROM rPPG</h1>
  <p><i>Towards Frictionless, Contactless Healthcare Screening via Chrominance-based Remote Photoplethysmography.</i></p>
</div>

---

## 📖 Project Overview
Opti-Screen is a **Remote Photoplethysmography (rPPG)** research platform that uses standard CMOS webcams to extract vital signs without physical contact. By detecting sub-perceptual changes in skin reflectance caused by the cardiac cycle, the system extracts heart rate, HRV, and stress metrics in real-time.

### 🎯 Core Analytical Capabilities
* **BPM (Heart Rate)**: Frequency-domain extraction via zero-padded 4096-point FFT for sub-BPM precision.
* **HRV (RMSSD)**: Root Mean Square of Successive Differences — Autonomic Nervous System (ANS) health indicator.
* **Baevsky Stress Index**: Physiological stress derived from Inter-Beat Interval (IBI) variance.
* **Environmental Guardrails**: Pseudo-Lux and Motion Delta monitoring for data integrity.

### 🆕 v4.0 Highlights
* **MediaPipe Face Mesh** — 468 landmarks for anatomically precise forehead ROI (Fitzpatrick I–VI inclusive)
* **Real PPG Waveform** — live signal visualization from actual CHROM output
* **Honest Confidence Reporting** — no inflated metrics; INSUFFICIENT_DATA returned when signal is poor
* **Calibration Progress Bar** — visible feedback during signal lock-on
* **Gauge-style HRV & Stress visuals** — instantly interpretable by non-technical viewers

---

## 🧬 Theoretical Framework: The CHROM Algorithm
Standard rPPG is susceptible to motion artifacts. Opti-Screen implements the **Chrominance-based (CHROM)** model (de Haan & Jeanne, 2013) to isolate the physiological pulse from environmental noise.

### 1. The Projection Formula
We normalize the RGB channels and project them into orthogonal chrominance vectors ($X$ and $Y$) that are blind to specular reflections:
* $$X = 3R_n - 2G_n$$
* $$Y = 1.5R_n + G_n - 1.5B_n$$

### 2. Alpha-Tuning & Filtering
To compensate for varied melanin concentrations and lighting conditions, we calculate an adaptive scalar ($\alpha$):
* $$Pulse = X - \left(\frac{\sigma(X)}{\sigma(Y)}\right)Y$$
The signal is then processed through a **4th-order Butterworth Bandpass Filter** (0.8Hz – 2.5Hz) to isolate human pulse frequencies (48–150 BPM).

---

## 🛡️ Data Integrity & Guard Rails
To prevent unreliable readings, the system enforces environmental thresholds:

| Metric | Safe Range | Purpose |
| :--- | :--- | :--- |
| **Pseudo-Lux** | 50 – 210 | Prevents sensor noise in low light or clipping in overexposure |
| **Motion Delta** | < 15 px | Ensures ROI stability for spectral analysis |
| **Face Detection** | Continuous | 4-second no-face timeout triggers session abort |
| **Calibration** | ~1 second | Minimum buffer fill before BPM output |

---

## 🛠️ Installation & Execution

```bash
# Clone the Repository
git clone https://github.com/User-s22/Opti-Screen.git
cd Opti-Screen

# Setup Environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Launch Dashboard
python app.py
```

The dashboard will be available at `http://localhost:5002`.

---

## 📂 Key Files

| File | Role |
| :--- | :--- |
| `app.py` | Flask orchestrator — MJPEG streaming, session management, REST API |
| `core/rppg.py` | CHROM signal processing engine — FFT, HRV (RMSSD), Baevsky SI |
| `core/camera.py` | MediaPipe Face Mesh detector with Haar cascade fallback |
| `static/js/dashboard.js` | Real-time UI — PPG waveform, gauges, session timers |
| `static/css/style.css` | Glassmorphic dark theme with micro-animations |
| `validate_dataset.py` | MAE validation tool for benchmarking accuracy |
| `analyze_video.py` | CLI tool for offline video analysis |

---

## ⚠️ Disclaimer
This project is for **research and educational purposes only**. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.