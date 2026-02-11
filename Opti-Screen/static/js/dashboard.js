// Dashboard JavaScript for real-time updates

// Chart.js configuration
let ppgChart;
let ppgData = [];
const MAX_DATA_POINTS = 150; // 5 seconds at 30fps

// Initialize Chart
function initChart() {
    const ctx = document.getElementById('ppgChart').getContext('2d');

    ppgChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(MAX_DATA_POINTS).fill(''),
            datasets: [{
                label: 'PPG Signal',
                data: ppgData,
                borderColor: '#00ffff',
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0 // Disable animation for real-time updates
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#00ffff'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Update dashboard with new data
function updateDashboard(data) {
    // CRITICAL: Stop polling if video ended (prevent zero drop)
    if (data.status === 'VIDEO_ENDED' && window.pollInterval) {
        console.log('[DASHBOARD] Video ended - stopping polling to lock results');
        clearInterval(window.pollInterval);
        window.pollInterval = null;

        // Show completion message
        const statusDiv = document.getElementById('uploadStatus');
        if (statusDiv) {
            statusDiv.className = 'upload-status success';
            statusDiv.textContent = '✓ Analysis Complete - Results Locked';
        }
    }

    // Update mode indicator with color coding
    const modeElement = document.getElementById('currentMode');
    modeElement.textContent = `MODE: ${data.mode || 'undefined'}`;

    // Color coding: Cyan for FACE, Orange for FINGER
    if (data.mode === 'FACE') {
        modeElement.style.color = '#00bcd4'; // Cyan
    } else if (data.mode === 'FINGER') {
        modeElement.style.color = '#ff9800'; // Orange
    } else {
        modeElement.style.color = '#ffffff'; // White for undefined
    }

    // Update metrics with proper decimal formatting
    if (data.mode === 'FACE') {
        // Round BPM to integer
        document.getElementById('bpmValue').textContent = data.bpm ? Math.round(data.bpm) : '--';
        document.getElementById('anemiaValue').textContent = '--';

        // Update health remark (appears when video ends)
        const remarkElement = document.getElementById('healthRemark');
        if (data.remark && data.remark !== '') {
            remarkElement.textContent = data.remark;
            remarkElement.style.display = 'block';

            // Color code based on remark
            if (data.remark.includes('Normal')) {
                remarkElement.style.color = '#00ff88';  // Green
            } else if (data.remark.includes('Bradycardia') || data.remark.includes('Tachycardia')) {
                remarkElement.style.color = '#ff9800';  // Orange
            } else {
                remarkElement.style.color = '#ffaa00';  // Yellow
            }
        } else {
            remarkElement.style.display = 'none';
        }
    } else {
        document.getElementById('bpmValue').textContent = '--';
        document.getElementById('anemiaValue').textContent = data.anemia_ratio || '--';
    }

    // Round SQI to 1 decimal place
    document.getElementById('sqiValue').textContent = data.sqi ? parseFloat(data.sqi).toFixed(1) : '--';

    // Round stability to 1 decimal place
    document.getElementById('stabilityValue').textContent = data.stability ? parseFloat(data.stability).toFixed(1) : '--';

    // Update OHI with 2 decimal places
    const ohiValue = data.ohi || 0;
    document.getElementById('ohiValue').textContent = parseFloat(ohiValue).toFixed(2);
    document.getElementById('ohiStatus').textContent = data.classification || 'UNKNOWN';

    // Update OHI status styling
    const ohiStatus = document.getElementById('ohiStatus');
    ohiStatus.className = 'ohi-status';
    if (data.classification === 'OPTIMAL') {
        ohiStatus.classList.add('optimal');
    } else if (data.classification === 'MONITOR') {
        ohiStatus.classList.add('monitor');
    } else if (data.classification === 'LOW') {
        ohiStatus.classList.add('low');
    }

    // Update OHI bar
    const ohiFill = document.getElementById('ohiFill');
    ohiFill.style.width = `${(ohiValue / 10) * 100}%`;

    // Update warnings
    const warningsList = document.getElementById('warningsList');
    if (data.warnings && data.warnings.length > 0) {
        warningsList.innerHTML = data.warnings.map(w =>
            `<p>⚠️ ${w}</p>`
        ).join('');
    } else {
        warningsList.innerHTML = '<p class="no-warnings">✓ No warnings</p>';
    }

    // Update PPG chart (simulate waveform with BPM)
    // In a real implementation, you'd get actual signal data from the backend
    if (data.mode === 'FACE' && data.bpm > 0) {
        const newValue = Math.sin(Date.now() / 100) * 50 + (data.bpm - 60);
        ppgData.push(newValue);
    } else {
        ppgData.push(0);
    }

    // Keep only last MAX_DATA_POINTS
    if (ppgData.length > MAX_DATA_POINTS) {
        ppgData.shift();
    }

    // Update chart
    ppgChart.data.datasets[0].data = ppgData;
    ppgChart.update('none'); // Update without animation
}

// Fetch status from server
async function fetchStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

// Mode toggle (placeholder - would need backend support)
document.addEventListener('DOMContentLoaded', () => {
    initChart();

    // Start fetching status every 1 second (store interval ID for later clearing)
    window.pollInterval = setInterval(fetchStatus, 1000);

    // Initial fetch
    fetchStatus();

    // Mode toggle button
    document.getElementById('modeToggle').addEventListener('click', async () => {
        try {
            const response = await fetch('/toggle_mode', {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Immediately update the mode display
                const modeElement = document.getElementById('currentMode');
                modeElement.textContent = `MODE: ${data.mode}`;

                // Update color
                if (data.mode === 'FACE') {
                    modeElement.style.color = '#00bcd4'; // Cyan
                } else if (data.mode === 'FINGER') {
                    modeElement.style.color = '#ff9800'; // Orange
                }

                // Show success message briefly
                const statusDiv = document.getElementById('uploadStatus');
                statusDiv.className = 'upload-status success';
                statusDiv.textContent = `✓ ${data.message}`;

                setTimeout(() => {
                    statusDiv.textContent = '';
                    statusDiv.className = 'upload-status';
                }, 2000);
            }
        } catch (error) {
            console.error('Error toggling mode:', error);
        }
    });

    // Video upload handler
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const fileInput = document.getElementById('videoFile');
        const statusDiv = document.getElementById('uploadStatus');

        if (!fileInput.files || fileInput.files.length === 0) {
            statusDiv.className = 'upload-status error';
            statusDiv.textContent = '❌ Please select a video file';
            return;
        }

        const formData = new FormData();
        formData.append('video', fileInput.files[0]);

        // Show loading status
        statusDiv.className = 'upload-status loading';
        statusDiv.textContent = '⏳ Uploading video...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                statusDiv.className = 'upload-status success';
                statusDiv.textContent = `✓ ${data.message}`;

                // Reload video feed after 1 second
                setTimeout(() => {
                    location.reload();
                }, 1000);
            } else {
                statusDiv.className = 'upload-status error';
                statusDiv.textContent = `❌ ${data.error || 'Upload failed'}`;
            }
        } catch (error) {
            statusDiv.className = 'upload-status error';
            statusDiv.textContent = `❌ Error: ${error.message}`;
        }
    });

    // Reset camera handler
    document.getElementById('resetCamera').addEventListener('click', async () => {
        const statusDiv = document.getElementById('uploadStatus');

        statusDiv.className = 'upload-status loading';
        statusDiv.textContent = '⏳ Clearing video...';

        try {
            const response = await fetch('/reset_camera', {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok && data.success) {
                statusDiv.className = 'upload-status success';
                statusDiv.textContent = '✓ Video cleared - ready for new upload';

                // Reload page after 1 second
                setTimeout(() => {
                    location.reload();
                }, 1000);
            } else {
                statusDiv.className = 'upload-status error';
                statusDiv.textContent = '❌ Reset failed';
            }
        } catch (error) {
            statusDiv.className = 'upload-status error';
            statusDiv.textContent = `❌ Error: ${error.message}`;
        }
    });
});

// Update chart more frequently for smooth animation (30fps)
setInterval(() => {
    if (ppgChart && ppgData.length > 0) {
        ppgChart.update('none');
    }
}, 33); // ~30 FPS
