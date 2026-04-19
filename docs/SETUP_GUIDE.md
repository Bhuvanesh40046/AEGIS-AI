# AEGIS.AI — Phase 1: Helmet Detection Setup Guide

## Overview

This module detects workers in a camera feed, tracks them with persistent IDs
using ByteTrack, and identifies who is NOT wearing a helmet using IoU-based
association. Voice alerts are triggered for violators with a cooldown system.

---

## Step 1: Environment Setup

### 1a. Check your GPU
Open a terminal and run:
```bash
nvidia-smi
```
You should see your 4GB GPU listed. Note the CUDA version shown (e.g., 12.1).

### 1b. Create a Python virtual environment
```bash
python -m venv aegis_env

# Windows:
aegis_env\Scripts\activate

# Linux/Mac:
source aegis_env/bin/activate
```

### 1c. Install PyTorch with GPU support
Go to https://pytorch.org/get-started/locally/ and select your CUDA version.

Example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 1d. Verify GPU is detected by PyTorch
```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print `True` and your GPU name.

### 1e. Install project dependencies
```bash
pip install -r requirements.txt
```

---

## Step 2: Get the Dataset

### Option A — Roboflow (Recommended, Fastest)

1. Go to [Roboflow Universe](https://universe.roboflow.com)
2. Search for **"hard hat detection"** or **"safety helmet detection"**
3. Recommended datasets:
   - **"Hard Hat Workers"** by Roboflow (5000+ images, 3 classes)
   - **"Construction Safety"** datasets
4. Click **"Download Dataset"**
5. Select format: **YOLOv8**
6. Download and extract to: `./datasets/helmet_dataset/`

### Option B — Create Your Own Dataset

1. Collect 500+ images of workers with and without helmets
   - Different angles, lighting, distances
   - Groups of workers (some with, some without)
   - Various helmet colors and types
2. Upload to [Roboflow](https://app.roboflow.com) (free tier allows 1000 images)
3. Draw bounding boxes around:
   - `helmet` — the helmet on a worker's head
   - `no_helmet` — a worker's head WITHOUT a helmet
   - `person` — the full body of each worker
4. Export in YOLOv8 format

### Important: Class Mapping

After downloading, open `data.yaml` in your dataset folder and check the class names.
Common variations:

```
# If your dataset has these classes:
names:
  0: 'helmet'         → maps to Config.HELMET_CLASS_IDS = [0]
  1: 'no_helmet'      → maps to Config.NO_HELMET_CLASS_IDS = [1]
  2: 'person'         → maps to Config.PERSON_CLASS_IDS = [2]
```

```
# Some datasets use different names:
names:
  0: 'hard-hat'       → Update Config.HELMET_CLASS_IDS = [0]
  1: 'head'           → Update Config.NO_HELMET_CLASS_IDS = [1]
  2: 'person'         → Config.PERSON_CLASS_IDS = [2]
```

**You MUST update the class IDs in `helmet_detector.py` → Config section
to match YOUR dataset's class order.** This is the most common source of errors.

### Verify Dataset Structure
```
datasets/
└── helmet_dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/    (should have .jpg/.png files)
    │   └── labels/    (should have .txt files with same names)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/          (optional)
        ├── images/
        └── labels/
```

---

## Step 3: Train the Model

```bash
python train_helmet_model.py --epochs 50 --batch 16 --size s --device 0
```

### Parameter Guide:
| Parameter    | Recommendation for 4GB GPU      |
|-------------|----------------------------------|
| `--size`    | `s` (small) — best speed/accuracy balance |
| `--batch`   | `8` or `16` — reduce to 8 if you get CUDA out of memory |
| `--epochs`  | `50` — increase to 100 for better accuracy |
| `--img-size`| `640` (default) — reduce to 416 if memory issues |

### If you get "CUDA out of memory":
```bash
# Reduce batch size
python train_helmet_model.py --batch 8

# Or use nano model (faster, slightly less accurate)
python train_helmet_model.py --size n --batch 16

# Or reduce image size
python train_helmet_model.py --batch 16 --img-size 416
```

### Training will take:
- ~30–60 minutes for 50 epochs with 5000 images on a 4GB GPU
- You'll see mAP (mean Average Precision) improving each epoch
- Training auto-stops if no improvement for 20 epochs

### After Training:
```bash
# Copy the best model to your project root
cp runs/detect/train/weights/best.pt ./best.pt
```

Training plots (loss curves, confusion matrix, PR curve) are saved in:
`runs/detect/train/`

---

## Step 4: Run the Detection System

### Basic Run (default webcam):
```bash
python helmet_detector.py
```

### With External USB Webcam:
```bash
python helmet_detector.py --camera 1
```
(Try 0, 1, 2 to find your webcam index)

### With Custom Model Path:
```bash
python helmet_detector.py --model ./best.pt --camera 0
```

### With Debug Mode (shows head regions and IoU boxes):
```bash
python helmet_detector.py --debug
```

### Test Pipeline Without Trained Model (COCO fallback):
```bash
python helmet_detector.py --coco-fallback
```
This uses the standard YOLOv8 model to detect persons only (no helmet detection).
Useful for testing if camera, tracking, and UI work before you train your model.

---

## Step 5: Keyboard Controls (While Running)

| Key | Action                                    |
|-----|-------------------------------------------|
| Q   | Quit the application                      |
| D   | Toggle debug mode (show head regions)     |
| S   | Save screenshot to current folder         |
| M   | Mute / Unmute voice alerts                |

---

## How the Detection Logic Works

```
For each frame:
  1. YOLO detects all objects → persons, helmets, no_helmets (bare heads)
  2. ByteTrack assigns persistent ID to each person (Person-1, Person-2, ...)
  3. For each person:
     a. Extract HEAD REGION = top 30% of person bounding box
     b. Check IoU of head region with all detected 'helmet' boxes
     c. Check IoU of head region with all detected 'no_helmet' boxes
     d. If helmet IoU > 0.15 → person is wearing helmet → GREEN box
     e. If no_helmet IoU > 0.15 → person NOT wearing helmet → RED box
     f. If neither matches → conservative: flag as no helmet → RED box
  4. For each flagged person:
     a. Check cooldown: was this person alerted in last 15 seconds?
     b. If no → generate voice alert + log it
     c. If yes → skip (don't repeat)
```

---

## Troubleshooting

### "Model not found: best.pt"
You haven't trained the model yet. Either:
- Run `python train_helmet_model.py` first, OR
- Use `--coco-fallback` flag to test with COCO model, OR
- Download a pre-trained model from Roboflow

### "Cannot open camera 0"
- Check if another app (Zoom, Teams) is using the camera
- Try different indices: `--camera 1`, `--camera 2`
- On Linux, check permissions: `ls -la /dev/video*`

### "CUDA out of memory" during inference
- Close other GPU-using apps (browser with hardware acceleration, etc.)
- Use YOLOv8n (nano) instead of YOLOv8s: train with `--size n`

### Detections are inaccurate
- Train for more epochs (100+)
- Use a larger dataset (2000+ images minimum)
- Ensure your training images match your deployment environment
  (similar lighting, camera angle, distance)
- Check that class IDs in Config match your data.yaml

### Voice alerts not working
- Install pyttsx3: `pip install pyttsx3`
- On Linux, you may need: `sudo apt install espeak`
- Check that laptop speakers/volume are not muted

### FPS is very low (< 5)
- Use YOLOv8n instead of YOLOv8s
- Process every 2nd frame (modify in helmet_detector.py)
- Close other applications using GPU
- Export model to ONNX: `python train_helmet_model.py --export-onnx`

---

## Files in This Module

| File                    | Purpose                                     |
|------------------------|---------------------------------------------|
| helmet_detector.py     | Main detection + tracking + alerts          |
| train_helmet_model.py  | Train YOLOv8 on helmet dataset              |
| requirements.txt       | Python dependencies                         |
| SETUP_GUIDE.md         | This file                                   |

---

## Integration with Other Phases

This module exposes a `get_status()` method that returns:
```json
{
  "module": "helmet_detection",
  "total_workers": 5,
  "violations": 1,
  "violation_details": [
    {"person_id": 3, "confidence": 0.87, "box": [100, 50, 300, 400]}
  ],
  "timestamp": "2026-03-28T14:32:05"
}
```

In Phase 4 (Orchestrator), this status dict is consumed by the central
priority queue alongside Zone Agent and Environmental Agent outputs.
