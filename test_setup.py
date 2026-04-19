"""
=============================================================================
 AEGIS.AI — System Check / Quick Test
=============================================================================
 Run this FIRST to verify everything is installed and working:
   python test_setup.py
 
 Checks:
   1. Python version
   2. Required libraries installed
   3. GPU availability + VRAM
   4. Camera accessibility
   5. TTS engine working
   6. YOLO model download + basic inference
=============================================================================
"""

import sys
import os

def check(name, condition, fix=""):
    status = "PASS" if condition else "FAIL"
    icon = "✓" if condition else "✗"
    print(f"  [{icon}] {name}")
    if not condition and fix:
        print(f"      Fix: {fix}")
    return condition

def main():
    print("=" * 60)
    print("  AEGIS.AI — System Check")
    print("=" * 60)
    all_pass = True
    
    # ── 1. Python Version ────────────────────────────────────────────
    print("\n[1/6] Python Version")
    ver = sys.version_info
    ok = ver.major == 3 and ver.minor >= 8
    all_pass &= check(f"Python {ver.major}.{ver.minor}.{ver.micro}", ok,
                       "Need Python 3.8+. Download from python.org")
    
    # ── 2. Required Libraries ────────────────────────────────────────
    print("\n[2/6] Required Libraries")
    
    libs = {
        'cv2': ('opencv-python', 'pip install opencv-python'),
        'numpy': ('numpy', 'pip install numpy'),
        'ultralytics': ('ultralytics', 'pip install ultralytics'),
        'yaml': ('PyYAML', 'pip install PyYAML'),
    }
    
    for module, (name, fix) in libs.items():
        try:
            __import__(module)
            all_pass &= check(f"{name} installed", True)
        except ImportError:
            all_pass &= check(f"{name} installed", False, fix)
    
    # Optional libs
    try:
        import pyttsx3
        check("pyttsx3 installed (voice alerts)", True)
    except ImportError:
        check("pyttsx3 installed (voice alerts)", False,
              "pip install pyttsx3 — voice alerts will be disabled without it")
    
    try:
        import lap
        check("lap installed (ByteTrack dependency)", True)
    except ImportError:
        check("lap installed (ByteTrack dependency)", False,
              "pip install lap")
    
    # ── 3. GPU Check ─────────────────────────────────────────────────
    print("\n[3/6] GPU Availability")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        all_pass &= check(f"PyTorch installed (v{torch.__version__})", True)
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            check(f"CUDA available: {gpu_name} ({vram:.1f} GB)", True)
            
            if vram >= 4:
                check("VRAM sufficient (>= 4GB)", True)
                print(f"      Recommended: YOLOv8s with batch=16")
            elif vram >= 2:
                check("VRAM limited (2-4GB)", True)
                print(f"      Recommended: YOLOv8n with batch=8")
            else:
                check("VRAM very low (<2GB)", False,
                      "Use CPU mode or YOLOv8n with batch=4")
        else:
            check("CUDA available", False,
                  "Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
            print("      System will fall back to CPU (slower but works)")
    except ImportError:
        all_pass &= check("PyTorch installed", False,
                          "pip install torch torchvision")
    
    # ── 4. Camera Check ──────────────────────────────────────────────
    print("\n[4/6] Camera Accessibility")
    
    try:
        import cv2
        
        # Try camera indices 0, 1, 2
        found_cameras = []
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found_cameras.append((idx, w, h))
                cap.release()
            else:
                cap.release()
        
        if found_cameras:
            for idx, w, h in found_cameras:
                check(f"Camera {idx}: {w}x{h}", True)
            print(f"      Use --camera {found_cameras[0][0]} for default")
        else:
            check("No camera found", False,
                  "Connect a webcam or check if another app is using it")
    except Exception as e:
        check(f"Camera check failed: {e}", False)
    
    # ── 5. TTS Check ─────────────────────────────────────────────────
    print("\n[5/6] Text-to-Speech Engine")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        check(f"TTS engine initialized ({len(voices)} voices)", True)
        
        # Quick test speak
        engine.say("System check complete")
        engine.runAndWait()
        check("TTS audio output working", True)
        del engine
    except Exception as e:
        check(f"TTS test: {e}", False,
              "pip install pyttsx3. On Linux: sudo apt install espeak")
    
    # ── 6. YOLO Quick Test ───────────────────────────────────────────
    print("\n[6/6] YOLOv8 Quick Test")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Download and test with nano model (smallest, ~6MB)
        print("      Downloading YOLOv8n (one-time, ~6MB)...")
        model = YOLO("yolov8n.pt")
        
        # Create a dummy image and run inference
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        
        check("YOLOv8 inference working", True)
        
        # Test tracking
        results_track = model.track(dummy, verbose=False, persist=True)
        check("ByteTrack tracking working", True)
        
        print(f"      Model classes: {model.names}")
        
    except Exception as e:
        check(f"YOLOv8 test failed: {e}", False,
              "pip install ultralytics")
    
    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_pass:
        print("  ALL CHECKS PASSED — Ready to go!")
        print()
        print("  Next steps:")
        print("  1. Get dataset (see SETUP_GUIDE.md, Step 2)")
        print("  2. Train: python train_helmet_model.py")
        print("  3. Run:   python helmet_detector.py")
        print()
        print("  Or test pipeline now (no training needed):")
        print("  python helmet_detector.py --coco-fallback")
    else:
        print("  SOME CHECKS FAILED — Fix the issues above first")
    print("=" * 60)


if __name__ == "__main__":
    main()
