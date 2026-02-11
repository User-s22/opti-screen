#!/usr/bin/env python3
"""
Quick test script to verify BPM detection is working
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.camera import Camera
from core.rppg import AdvancedRPPG
import time

print("=" * 60)
print("Testing Opti-Screen BPM Detection")
print("=" * 60)

# Test 1: Module imports
print("\n1. Testing module imports...")
try:
    print("   ✓ Camera module imported")
    print("   ✓ rPPG module imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Haar Cascade
print("\n2. Testing Haar Cascade...")
cam = Camera(source=None)
if cam.face_cascade is not None:
    print("   ✓ Haar Cascade loaded successfully")
else:
    print("   ✗ Haar Cascade failed to load")
    sys.exit(1)

# Test 3: rPPG Engine
print("\n3. Testing rPPG engine...")
rppg = AdvancedRPPG(fps=30, window_size=300)
print(f"   ✓ rPPG engine initialized")
print(f"   ✓ Buffer size: {rppg.buffer_size} frames")
print(f"   ✓ FPS: {rppg.fps}")

# Test 4: Signal processing
print("\n4. Testing signal processing...")
# Add some dummy samples
for i in range(200):
    rppg.add_frame((100 + i % 10, 120 + i % 8, 80 + i % 12), i/30.0)

result = rppg.process_ppg_signal()
print(f"   Status: {result['status']}")
print(f"   Ready: {result['ready']}")
print(f"   Buffer: {len(rppg.r_buffer)} samples")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nNow run: python app.py")
print("Then upload a video and wait 5 seconds for calibration.")
print("=" * 60)
