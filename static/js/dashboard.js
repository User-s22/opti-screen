// ─────────────────────────────────────────────────────────────────
// OPTI-SCREEN  |  Dashboard JS  v4 — Production Upgrade
// ─────────────────────────────────────────────────────────────────
'use strict';

// ── Chart ──────────────────────────────────────────────────────────
let ppgChart;
const MAX_PPG_POINTS = 150;

function initChart() {
    const ctx = document.getElementById('ppgChart').getContext('2d');
    ppgChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(MAX_PPG_POINTS).fill(''),
            datasets: [{
                label: 'PPG Signal',
                data: [],
                borderColor: '#00e5ff',
                backgroundColor: 'rgba(0,229,255,0.07)',
                borderWidth: 1.8,
                tension: 0.45,
                pointRadius: 0,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    grid: { color: 'rgba(0,229,255,0.07)' },
                    ticks: { color: '#5a6a82', font: { size: 10 } }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// ── Helpers ────────────────────────────────────────────────────────
function setValue(id, text) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.textContent !== String(text)) {
        el.textContent = text;
        el.classList.remove('value-pop');
        void el.offsetWidth;
        el.classList.add('value-pop');
    }
}

// ── Session state ──────────────────────────────────────────────────
let sessionMode = null;   // 'live' | 'video'
let sessionActive = false;
let sessionStartMs = 0;
let faultTicks = 0;
let noFaceTicks = 0;
let sessionEnded = false;

const LIVE_DURATION_SEC = 30;
const FAULT_THRESHOLD = 1;
const NO_FACE_THRESHOLD = 8;

// ── Modal helpers ──────────────────────────────────────────────────
function stopStream() {
    if (window.pollInterval) { clearInterval(window.pollInterval); window.pollInterval = null; }
    const vf = document.getElementById('videoFeed');
    if (vf) { vf.src = ''; vf.style.display = 'none'; }
    document.querySelectorAll('.metric-card, .ohi-card, .classification-card').forEach(c => c.classList.remove('live-active'));
    sessionEnded = true;
}

async function showSuccessModal() {
    stopStream();
    try {
        const resp = await fetch('/session_summary');
        const s = await resp.json();

        // Handle INSUFFICIENT_DATA
        if (s.remark === 'INSUFFICIENT_DATA' || s.avg_bpm === null) {
            document.getElementById('insufficient-modal').style.display = 'flex';
            return;
        }

        document.getElementById('s-avg-bpm').textContent = s.avg_bpm > 0 ? s.avg_bpm : '--';
        document.getElementById('s-min-bpm').textContent = s.min_bpm > 0 ? s.min_bpm : '--';
        document.getElementById('s-max-bpm').textContent = s.max_bpm > 0 ? s.max_bpm : '--';
        document.getElementById('s-hrv').textContent = s.hrv > 0 ? s.hrv : '--';
        document.getElementById('s-stress').textContent = s.stress_index > 0 ? s.stress_index : '--';
        document.getElementById('s-stability').textContent = s.stability_pct != null ? s.stability_pct + '%' : '--';

        const cls = document.getElementById('s-classification');
        cls.textContent = s.classification || '--';
        cls.style.color = classificationColor(s.classification);

        const rem = document.getElementById('s-remark');
        rem.textContent = s.remark || '';
    } catch (e) { console.error('Summary fetch failed', e); }
    document.getElementById('success-modal').style.display = 'flex';
}

function showAbortModal(reason) {
    stopStream();
    const r = document.getElementById('abort-reason');
    if (r) r.textContent = reason || 'Unstable lighting or excessive motion detected.';
    document.getElementById('abort-modal').style.display = 'flex';
}

// ── Classification colour helper ──────────────────────────────────
function classificationColor(cls) {
    switch (cls) {
        case 'NORMAL': return '#00ff88';
        case 'MONITOR': return '#ffab40';
        case 'BRADYCARDIA': return '#64b5f6';
        case 'TACHYCARDIA': return '#ff5252';
        default: return '#5a6a82';
    }
}

function classificationLabel(cls) {
    switch (cls) {
        case 'NORMAL': return 'Normal resting heart rate';
        case 'MONITOR': return 'Slightly elevated — keep monitoring';
        case 'BRADYCARDIA': return 'Below normal — bradycardia range';
        case 'TACHYCARDIA': return 'Above normal — tachycardia range';
        default: return 'Awaiting signal…';
    }
}

// ── Main update function ───────────────────────────────────────────
function updateDashboard(data) {
    if (sessionEnded) return;

    const lux = typeof data.estimated_lux === 'number' ? data.estimated_lux : 0;
    const motion = typeof data.motion_delta === 'number' ? data.motion_delta : 0;
    const calibrated = data.calibration_done === true;

    // ── Calibration progress bar ────────────────────────────────
    const calBar = document.getElementById('calibrationBar');
    const calFill = document.getElementById('calibrationFill');
    const calLabel = document.getElementById('calibrationLabel');
    if (calBar) {
        if (calibrated) {
            calBar.style.display = 'none';
        } else {
            calBar.style.display = 'block';
            const progress = data.calibration_progress || 0;
            if (calFill) calFill.style.width = `${Math.min(100, progress)}%`;
            if (calLabel) {
                const faceOk = data.face_detected === true;
                calLabel.textContent = faceOk
                    ? `⏳ Calibrating signal… ${progress}% — please stay still`
                    : '👁️ Looking for face… please center your face';
            }
        }
    }

    // ── LIVE SESSION GUARDS ──────────────────────────────────────
    if (sessionMode === 'live') {
        const banner = document.getElementById('sessionBanner');
        const timerEl = document.getElementById('sessionTimer');
        const statusEl = document.getElementById('sessionStatus');
        if (banner) banner.style.display = 'flex';

        const faceOk = data.face_detected === true;

        if (!faceOk) {
            noFaceTicks++;
            if (statusEl) statusEl.textContent = '👁️ No face detected — center your face';
            if (timerEl) timerEl.textContent = '';
            if (noFaceTicks >= NO_FACE_THRESHOLD) {
                showAbortModal('Face not detected for 4 seconds. Ensure your face is centered and the room is well-lit.');
                return;
            }
        } else {
            noFaceTicks = 0;
        }

        if (!calibrated) {
            if (faceOk && statusEl) statusEl.textContent = '⏳ Calibrating… please stay still';
            if (timerEl) timerEl.textContent = '';
        } else {
            if (!sessionActive) {
                sessionActive = true;
                sessionStartMs = Date.now();
                faultTicks = 0;
                document.querySelectorAll('.metric-card, .classification-card').forEach(c => c.classList.add('live-active'));
            }

            const elapsed = (Date.now() - sessionStartMs) / 1000;
            const remain = Math.max(0, Math.ceil(LIVE_DURATION_SEC - elapsed));

            if (statusEl) statusEl.textContent = '🔴 Recording…';
            if (timerEl) timerEl.textContent = `${remain}s remaining`;

            if (elapsed >= LIVE_DURATION_SEC) {
                showSuccessModal();
                return;
            }

            const badLux = lux < 50 || lux > 210;
            const badMotion = motion > 15;
            if (badLux || badMotion) {
                faultTicks++;
            } else {
                faultTicks = 0;
            }
            if (faultTicks >= FAULT_THRESHOLD) {
                const why = badLux
                    ? `Poor lighting detected (lux=${lux}). Please improve lighting and try again.`
                    : `Excessive head movement detected (delta=${motion}px). Please stay still and try again.`;
                showAbortModal(why);
                return;
            }
        }
    }

    // ── VIDEO MODE: detect when video ends ───────────────────────
    if (sessionMode === 'video' && data.status === 'VIDEO_ENDED') {
        showSuccessModal();
        return;
    }

    // ── UI Updates ───────────────────────────────────────────────

    // BPM
    const bpmNum = data.bpm ? Math.round(data.bpm) : 0;
    setValue('bpmValue', bpmNum > 0 ? bpmNum : '--');
    const bpmBar = document.getElementById('bpmBar');
    if (bpmBar) bpmBar.style.width = `${Math.min(100, Math.max(0, (bpmNum - 40) / 1.2))}%`;

    // Classification badge
    const badge = document.getElementById('classificationBadge');
    const badgeSub = document.getElementById('classificationSub');
    if (badge) {
        const cls = data.classification || 'UNKNOWN';
        badge.textContent = cls;
        badge.style.color = classificationColor(cls);
        badge.style.borderColor = classificationColor(cls);
        badge.style.background = classificationColor(cls) + '15';
    }
    if (badgeSub) badgeSub.textContent = classificationLabel(data.classification);

    // HRV gauge
    const hrvVal = data.hrv > 0 ? parseFloat(data.hrv).toFixed(1) : '--';
    setValue('hrvValue', hrvVal);
    const hrvGauge = document.getElementById('hrvGauge');
    if (hrvGauge) {
        const pct = Math.min(100, (parseFloat(data.hrv) || 0) / 128 * 100);
        hrvGauge.style.width = `${pct}%`;
    }

    // Stress gauge
    const stressVal = data.stress_index > 0 ? parseFloat(data.stress_index).toFixed(1) : '--';
    setValue('stressValue', stressVal);
    const stressGauge = document.getElementById('stressGauge');
    if (stressGauge) {
        const pct = Math.min(100, (parseFloat(data.stress_index) || 0));
        stressGauge.style.width = `${pct}%`;
    }

    // Stability / Signal Quality
    const stabEl = document.getElementById('stabilityValue');
    if (stabEl) {
        stabEl.textContent = data.stability_indicator || '--';
        stabEl.style.color = data.stability_indicator === 'HIGH' ? '#00ff88'
            : data.stability_indicator === 'MEDIUM' ? '#ffaa00' : '#ff4444';
    }
    const confLabel = document.getElementById('confidenceLabel');
    if (confLabel) {
        const conf = parseInt(data.confidence) || 0;
        confLabel.textContent = `Confidence: ${conf}%`;
    }

    // Lux & Motion
    setValue('luxValue', lux || '--');
    setValue('motionValue', motion || '--');

    // Warnings
    const wl = document.getElementById('warningsList');
    if (wl) {
        if (data.warnings && data.warnings.length > 0) {
            wl.innerHTML = data.warnings.map(w =>
                `<p style="color:#ffab40;border-left-color:#ffab40;background:rgba(255,171,64,0.08)">⚠️ ${w}</p>`
            ).join('');
            wl.closest('.card').style.borderColor = 'rgba(255,82,82,0.5)';
        } else {
            const luxLine = lux > 0 ? ` · Lux: ${lux}` : '';
            wl.innerHTML = `<p class="no-warnings">✓ All clear${luxLine}</p>`;
            wl.closest('.card').style.borderColor = '';
        }
    }

    // Health remark
    const rem = document.getElementById('healthRemark');
    if (rem) {
        if (data.remark) { rem.textContent = data.remark; rem.style.display = 'block'; }
        else rem.style.display = 'none';
    }

    // ── Real PPG waveform ────────────────────────────────────────
    if (ppgChart) {
        const ppg = data.ppg_signal;
        if (ppg && ppg.length > 0) {
            // Use real signal data from the backend
            const slice = ppg.slice(-MAX_PPG_POINTS);
            ppgChart.data.labels = Array(slice.length).fill('');
            ppgChart.data.datasets[0].data = slice;
        } else if (bpmNum > 0) {
            // Fallback: synthetic sine if no signal data yet
            let currentData = ppgChart.data.datasets[0].data || [];
            currentData.push(Math.sin(Date.now() / 95) * 45 + (bpmNum - 60) * 0.3);
            if (currentData.length > MAX_PPG_POINTS) currentData.shift();
            ppgChart.data.labels = Array(currentData.length).fill('');
            ppgChart.data.datasets[0].data = currentData;
        }
        ppgChart.update('none');
    }
}

// ── Polling ────────────────────────────────────────────────────────
async function fetchStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        updateDashboard(data);
    } catch (e) {
        console.warn('[POLL] fetch error:', e.message);
    }
}

function startPolling() {
    if (window.pollInterval) clearInterval(window.pollInterval);
    window.pollInterval = setInterval(fetchStatus, 400);
    fetchStatus();
}

// ── Show dashboard (hide landing) ─────────────────────────────────
function enterDashboard(mode) {
    sessionMode = mode;
    document.getElementById('landing').style.display = 'none';
    document.getElementById('dashboard').style.display = 'flex';
    const label = document.getElementById('feedLabel');
    if (mode === 'video' && label) label.textContent = '🎞️ Video Analysis Feed';

    const vf = document.getElementById('videoFeed');
    if (vf) {
        vf.src = '/video_feed';
        vf.style.display = 'block';
    }

    initChart();
    startPolling();
}

// ── DOMContentLoaded ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

    document.getElementById('backBtn')?.addEventListener('click', () => location.reload());

    // ── LIVE CAMERA ──────────────────────────────────────────────
    document.getElementById('startLiveBtn')?.addEventListener('click', async () => {
        const btn = document.getElementById('startLiveBtn');
        btn.textContent = '⏳ Starting…';
        btn.disabled = true;
        try {
            const res = await fetch('/start_webcam', { method: 'POST' });
            const data = await res.json();
            if (res.ok && data.success) {
                enterDashboard('live');
            } else {
                btn.textContent = '❌ Camera Error — Retry';
                btn.disabled = false;
            }
        } catch (e) {
            btn.textContent = '❌ Error — Retry';
            btn.disabled = false;
        }
    });

    // ── VIDEO UPLOAD ─────────────────────────────────────────────
    document.getElementById('uploadForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('videoFile');
        const statusEl = document.getElementById('uploadStatus');
        if (!fileInput?.files?.length) {
            statusEl.className = 'upload-status error';
            statusEl.textContent = '❌ Please select a video file';
            return;
        }
        const fd = new FormData();
        fd.append('video', fileInput.files[0]);
        statusEl.className = 'upload-status loading';
        statusEl.textContent = '⏳ Uploading…';
        try {
            const res = await fetch('/upload', { method: 'POST', body: fd });
            const data = await res.json();
            if (res.ok && data.success) {
                statusEl.className = 'upload-status success';
                statusEl.textContent = `✓ ${data.message} — starting analysis…`;
                setTimeout(() => enterDashboard('video'), 800);
            } else {
                statusEl.className = 'upload-status error';
                statusEl.textContent = `❌ ${data.error || 'Upload failed'}`;
            }
        } catch (err) {
            statusEl.className = 'upload-status error';
            statusEl.textContent = `❌ ${err.message}`;
        }
    });
});
