#!/usr/bin/env python3
"""
Camera Diagnostic Tool
Tests camera formats and measures raw capture speed
"""

import cv2
import time
import sys

def test_format(width, height, fourcc_str, fps_target=30):
    """Test a specific camera format and measure FPS"""
    print(f"\nTesting {fourcc_str} @ {width}x{height} @ {fps_target} fps...")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Check what was actually set
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fcc = ''.join([chr((actual_fourcc >> (8*i)) & 0xFF) for i in range(4)])
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Requested: {fourcc_str} @ {width}x{height} @ {fps_target} fps")
    print(f"  Actually got: {actual_fcc} @ {actual_w}x{actual_h} @ {actual_fps} fps")

    if actual_fcc != fourcc_str:
        print(f"  ⚠️  WARNING: Camera fell back to {actual_fcc}!")
        cap.release()
        return None

    # Warmup
    print("  Warming up (10 frames)...")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("  ❌ Failed to read frame during warmup")
            cap.release()
            return None

    # Measure capture speed
    N = 120
    print(f"  Measuring capture speed ({N} frames)...")
    start_time = time.time()
    successful_reads = 0

    for i in range(N):
        t0 = time.time()
        ret, frame = cap.read()
        t1 = time.time()

        if ret:
            successful_reads += 1
            read_time_ms = (t1 - t0) * 1000
            if i == 0 or i == N - 1 or read_time_ms > 200:
                print(f"    Frame {i}: {read_time_ms:.1f} ms")
        else:
            print(f"    Frame {i}: FAILED")

    elapsed = time.time() - start_time
    measured_fps = successful_reads / elapsed

    print(f"\n  Results:")
    print(f"    Successful frames: {successful_reads}/{N}")
    print(f"    Total time: {elapsed:.2f} seconds")
    print(f"    Measured FPS: {measured_fps:.1f}")
    print(f"    Average frame time: {(elapsed/successful_reads)*1000:.1f} ms")

    cap.release()
    return measured_fps

def main():
    print("=" * 70)
    print("CAMERA DIAGNOSTIC TOOL")
    print("=" * 70)

    # Test different configurations
    configs = [
        (640, 360, 'MJPG', 30),
        (640, 480, 'MJPG', 30),
        (320, 240, 'MJPG', 30),
        (640, 360, 'YUYV', 30),  # Will likely be slow
    ]

    results = []
    for width, height, fourcc, fps_target in configs:
        result = test_format(width, height, fourcc, fps_target)
        results.append((width, height, fourcc, result))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    for width, height, fourcc, fps in results:
        if fps:
            status = "✓ GOOD" if fps > 15 else "⚠️  SLOW"
            print(f"{fourcc} @ {width}x{height}: {fps:.1f} FPS - {status}")
        else:
            print(f"{fourcc} @ {width}x{height}: FAILED / NOT SUPPORTED")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    # Find best config
    mjpg_results = [(w, h, fps) for w, h, fc, fps in results if fc == 'MJPG' and fps]
    if mjpg_results:
        best = max(mjpg_results, key=lambda x: x[2])
        print(f"✓ Use MJPG @ {best[0]}x{best[1]} for best performance ({best[2]:.1f} FPS)")
    else:
        print("⚠️  MJPG not supported - camera will be SLOW!")
        print("   Consider using a different USB camera with MJPG support")

    print("=" * 70)

if __name__ == "__main__":
    main()
