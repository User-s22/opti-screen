#!/usr/bin/env python3
"""
Opti-Screen Round 2 - BPM Validation Tool
Runs analysis on a set of videos and compares against Ground Truth (GT) BPM.
Calculates Mean Absolute Error (MAE) for credibility.
"""
import os
import sys

# Suppress debug output from analyze_video for clean table
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

from analyze_video import analyze_video_complete

def run_validation(dataset):
    print("=" * 80)
    print("🫀 Opti-Screen - BPM Validation & MAE Evaluation")
    print("=" * 80)
    print(f"{'Video File':<30} | {'GT BPM':<10} | {'Our BPM':<10} | {'Error (MAE)':<15}")
    print("-" * 80)
    
    total_error = 0
    valid_count = 0
    
    for item in dataset:
        video_path = item['video']
        gt_bpm = item['gt_bpm']
        
        if not os.path.exists(video_path):
            print(f"{video_path[:28]:<30} | {'NOT FOUND':<10} | {'-':<10} | {'-':<15}")
            continue
            
        print(f"Processing {video_path}...", end='\r')
        
        # Suppress standard output of analyze_video_complete for clean table
        with HiddenPrints():
            results = analyze_video_complete(video_path)
        
        # Clear processing line
        sys.stdout.write("\033[K")
        
        if results['success']:
            our_bpm = results['bpm']
            error = abs(our_bpm - gt_bpm)
            total_error += error
            valid_count += 1
            print(f"{video_path[:28]:<30} | {gt_bpm:<10} | {our_bpm:<10.1f} | {error:.1f} BPM")
        else:
            print(f"{video_path[:28]:<30} | {gt_bpm:<10} | {'FAILED':<10} | {'-':<15}")
            
    print("-" * 80)
    if valid_count > 0:
        mae = total_error / valid_count
        print(f"Overall Mean Absolute Error (MAE): {mae:.2f} BPM across {valid_count} videos")
        if mae < 5.0:
            print("Status: 🟢 CLINICAL GRADE ACCURACY (MAE < 5 BPM)")
        elif mae < 10.0:
            print("Status: 🟡 CONSUMER GRADE ACCURACY (MAE < 10 BPM)")
        else:
            print("Status: 🔴 REQUIRES CALIBRATION (MAE >= 10 BPM)")
    else:
        print("No valid readings obtained from dataset.")
    print("=" * 80)

if __name__ == '__main__':
    # Default Dummy Dataset
    # Provide multiple video paths and their known ground truth BPM
    dataset = [
        {'video': 'uploads/Movie_on_16-02-26_at_4.19_PM.mov', 'gt_bpm': 75},
        {'video': 'uploads/missing.mp4', 'gt_bpm': 62},
    ]
    
    print("\nStarting BPM Validation Test...")
    run_validation(dataset)
