#!/usr/bin/env python3
"""
Test Advanced rPPG on Face Video
"""
import cv2
import numpy as np
from core.rppg import AdvancedRPPG, SkinToneNormalizer
from core.camera import Camera

print("="*70)
print("ADVANCED rPPG FACE ANALYSIS TEST")
print("="*70)

# Check if sample video exists
import os
video_path = "uploads/sample.mp4"

if not os.path.exists(video_path):
    print(f"\n⚠️  Video not found: {video_path}")
    print("Please upload a face video to test rPPG.\n")
    exit(1)

print(f"\nAnalyzing: {video_path}\n")

# Initialize
rppg = AdvancedRPPG(fps=30, window_size=300)
normalizer = SkinToneNormalizer()
camera = Camera(source=video_path)

frame_count = 0
bpm_readings = []

print("Processing frames...")
while True:
    frame_bytes, avg_r, avg_g, avg_b, mode = camera.get_frame()
    
    if frame_bytes is None:
        break
    
    # Normalize for skin tone
    r_norm, g_norm, b_norm = normalizer.normalize_rgb(avg_r, avg_g, avg_b)
    
    # Add to rPPG
    rppg.add_frame((r_norm, g_norm, b_norm))
    
    # Process with POS method
    result = rppg.process_ppg_signal(method='POS')
    
    if result['ready'] and result['bpm'] > 0:
        bpm_readings.append({
            'bpm': result['bpm'],
            'confidence': result['confidence']
        })
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"  Processed {frame_count} frames...")

print(f"\n✓ Completed: {frame_count} frames\n")

# Results
print("="*70)
print("RESULTS - Advanced rPPG (POS Method)")
print("="*70)

if len(bpm_readings) > 0:
    final_bpm = bpm_readings[-1]['bpm']
    final_conf = bpm_readings[-1]['confidence']
    
    all_bpms = [r['bpm'] for r in bpm_readings]
    avg_bpm = np.mean(all_bpms)
    std_bpm = np.std(all_bpms)
    
    print(f"\n🫀 HEART RATE (rPPG - POS Method)")
    print(f"   Final BPM: {final_bpm:.1f}")
    print(f"   Average BPM: {avg_bpm:.1f}")
    print(f"   Stability: ±{std_bpm:.1f} BPM")
    print(f"   Confidence: {final_conf:.1f}%")
    
    if 60 <= final_bpm <= 100:
        print(f"   Status: ✅ NORMAL (resting)")
    elif final_bpm < 60:
        print(f"   Status: ⚠️  BRADYCARDIA (low)")
    else:
        print(f"   Status: ⚠️  TACHYCARDIA (elevated)")
    
    print(f"\n📊 SIGNAL QUALITY")
    quality = rppg.get_signal_quality()
    print(f"   Quality Score: {quality:.1f}%")
    
    if quality > 70:
        print(f"   Assessment: ✅ EXCELLENT")
    elif quality > 50:
        print(f"   Assessment: ✅ GOOD")
    else:
        print(f"   Assessment: ⚠️  FAIR")
else:
    print("\n⚠️  No valid BPM readings detected")
    print("   - Ensure video shows a clear face")
    print("   - Check lighting conditions")
    print("   - Video should be at least 10 seconds")

print("\n" + "="*70)
print("✅ Advanced rPPG ready for hackathon!")
print("="*70 + "\n")
