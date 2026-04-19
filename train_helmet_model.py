"""
=============================================================================
 AEGIS.AI — Train Custom Helmet Detection Model (YOLOv8)
=============================================================================
 This script trains a YOLOv8 model to detect:
   - Class 0: 'helmet'      (worker wearing a helmet)
   - Class 1: 'no_helmet'   (worker's head without a helmet)
   - Class 2: 'person'      (full body of a worker)

 Dataset Setup:
   Option A — Download from Roboflow (RECOMMENDED for getting started):
     1. Go to https://roboflow.com
     2. Search for "helmet detection" or "hard hat detection"
     3. Popular datasets:
        - "Hard Hat Workers" dataset (~5000 images)
        - "Safety Helmet Detection" dataset
     4. Export in YOLOv8 format
     5. Extract to ./datasets/helmet_dataset/

   Option B — Create your own dataset:
     1. Collect images of workers with and without helmets
     2. Use Roboflow or LabelImg to annotate bounding boxes
     3. Label classes: helmet, no_helmet, person
     4. Export in YOLOv8 format

 Expected folder structure after setup:
   datasets/
   └── helmet_dataset/
       ├── data.yaml            ← class definitions + paths
       ├── train/
       │   ├── images/          ← training images
       │   └── labels/          ← YOLO format .txt annotations
       ├── valid/
       │   ├── images/          ← validation images
       │   └── labels/          ← validation annotations
       └── test/                ← (optional)
           ├── images/
           └── labels/

 Usage:
   python train_helmet_model.py
   python train_helmet_model.py --epochs 100 --batch 16
   python train_helmet_model.py --resume  # Resume interrupted training
=============================================================================
"""

import os
import argparse
import yaml


def create_dataset_yaml(dataset_path: str, num_classes: int = 3):
    """
    Create or verify the data.yaml file for the dataset.
    Adjust class names/count based on YOUR dataset.
    """
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if os.path.exists(yaml_path):
        print(f"[DATA] Found existing data.yaml: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        print(f"[DATA] Classes: {data.get('names', 'unknown')}")
        print(f"[DATA] Number of classes: {data.get('nc', 'unknown')}")
        return yaml_path
    
    # Create a template data.yaml
    print(f"[DATA] Creating template data.yaml at: {yaml_path}")
    
    data = {
        'path': os.path.abspath(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': {
            0: 'Person',
            1: 'head',
            2: 'helmet'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"[DATA] Template created. Edit class names if your dataset differs.")
    return yaml_path


def check_dataset(dataset_path: str) -> bool:
    """Verify dataset folder structure exists."""
    required = [
        os.path.join(dataset_path, "train", "images"),
        os.path.join(dataset_path, "train", "labels"),
        os.path.join(dataset_path, "valid", "images"),
        os.path.join(dataset_path, "valid", "labels"),
    ]
    
    all_ok = True
    for path in required:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path}")
        if not exists:
            all_ok = False
    
    if all_ok:
        train_imgs = len(os.listdir(os.path.join(dataset_path, "train", "images")))
        val_imgs = len(os.listdir(os.path.join(dataset_path, "valid", "images")))
        print(f"  Training images: {train_imgs}")
        print(f"  Validation images: {val_imgs}")
    
    return all_ok


def train(args):
    """Run YOLOv8 training."""
    from ultralytics import YOLO
    
    print("=" * 60)
    print("  AEGIS.AI — Helmet Detection Model Training")
    print("=" * 60)
    
    # ── Check dataset ───────────────────────────────────────────────────
    print(f"\n[1/4] Checking dataset: {args.dataset}")
    
    if not os.path.exists(args.dataset):
        print(f"\n[ERROR] Dataset folder not found: {args.dataset}")
        print("\nTo get started:")
        print("  1. Go to https://roboflow.com")
        print("  2. Search 'Hard Hat Workers' or 'Safety Helmet Detection'")
        print("  3. Export in 'YOLOv8' format")
        print(f"  4. Extract to: {args.dataset}")
        print("\n  Or create the folder structure manually:")
        print(f"  mkdir -p {args.dataset}/train/images")
        print(f"  mkdir -p {args.dataset}/train/labels")
        print(f"  mkdir -p {args.dataset}/valid/images")
        print(f"  mkdir -p {args.dataset}/valid/labels")
        return
    
    if not check_dataset(args.dataset):
        print("\n[ERROR] Dataset structure incomplete. See above.")
        return
    
    # ── Prepare data.yaml ───────────────────────────────────────────────
    print(f"\n[2/4] Preparing data configuration...")
    data_yaml = create_dataset_yaml(args.dataset)
    
    # ── Load base model ─────────────────────────────────────────────────
    print(f"\n[3/4] Loading base model: yolov8{args.size}.pt")
    
    if args.resume and os.path.exists("runs/detect/train/weights/last.pt"):
        print("  Resuming from last checkpoint...")
        model = YOLO("runs/detect/train/weights/last.pt")
    else:
        model = YOLO(f"yolov8{args.size}.pt")  # Downloads pretrained weights
    
    # ── Train ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.img_size}")
    print(f"  Model size: YOLOv8{args.size}")
    print(f"  Device: {'GPU' if args.device == '0' else 'CPU'}")
    print()
    
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        device=args.device,
        patience=20,               # Early stopping after 20 epochs no improvement
        save=True,
        save_period=10,            # Save checkpoint every 10 epochs
        plots=True,                # Generate training plots
        verbose=True,
        
        # ── Data Augmentation (improves robustness) ─────────────────────
        augment=True,
        hsv_h=0.015,              # Hue augmentation
        hsv_s=0.7,                # Saturation augmentation
        hsv_v=0.4,                # Value/brightness augmentation
        degrees=10,               # Rotation (±10°)
        translate=0.1,            # Translation
        scale=0.5,                # Scale augmentation
        flipud=0.0,               # No vertical flip (helmets are always on top)
        fliplr=0.5,               # Horizontal flip
        mosaic=1.0,               # Mosaic augmentation
        mixup=0.1,                # Mixup augmentation
    )
    
    # ── Post-training ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best model saved at: runs/detect/train/weights/best.pt")
    print(f"  Last model saved at: runs/detect/train/weights/last.pt")
    print(f"  Training plots: runs/detect/train/")
    print()
    print("  Next steps:")
    print("  1. Copy best.pt to your project folder:")
    print("     cp runs/detect/train/weights/best.pt ./best.pt")
    print("  2. Run the detection system:")
    print("     python helmet_detector.py --model best.pt")
    print("=" * 60)
    
    # ── Validate ────────────────────────────────────────────────────────
    print("\n  Running validation on best model...")
    best_model = YOLO("runs/detect/train/weights/best.pt")
    metrics = best_model.val(data=data_yaml)
    
    print(f"\n  Validation Results:")
    print(f"    mAP50:    {metrics.box.map50:.4f}")
    print(f"    mAP50-95: {metrics.box.map:.4f}")
    print(f"    Precision: {metrics.box.mp:.4f}")
    print(f"    Recall:    {metrics.box.mr:.4f}")
    
    # ── Export to ONNX (optional, for faster CPU inference) ─────────────
    if args.export_onnx:
        print("\n  Exporting to ONNX format...")
        best_model.export(format="onnx", dynamic=True, simplify=True)
        print("  ONNX model saved: runs/detect/train/weights/best.onnx")


def main():
    parser = argparse.ArgumentParser(
        description="AEGIS.AI — Train Helmet Detection Model"
    )
    parser.add_argument('--dataset', type=str, 
                        default='./datasets/helmet_dataset',
                        help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (reduce if GPU memory error)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--size', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n=nano, s=small, m=medium')
    parser.add_argument('--device', type=str, default='0',
                        help='Device: 0=GPU, cpu=CPU')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from last checkpoint')
    parser.add_argument('--export-onnx', action='store_true', default=False,
                        help='Export best model to ONNX after training')
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
