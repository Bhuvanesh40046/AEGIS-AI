import os
import shutil
import random

random.seed(42)

img_dir = "D:/hack/train/images"
lbl_dir = "D:/hack/train/labels"
val_img = "D:/hack/valid/images"
val_lbl = "D:/hack/valid/labels"

# Step 1: Move any existing valid images back to train
if os.path.exists(val_img):
    for f in os.listdir(val_img):
        if f.endswith((".jpg", ".png", ".jpeg")):
            src = os.path.join(val_img, f)
            dst = os.path.join(img_dir, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
    print("Moved validation images back to train.")

# Step 2: Create valid folders
os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

# Step 3: Get all training images
imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(imgs)
split = int(len(imgs) * 0.85)

print(f"Total images found: {len(imgs)}")
print(f"Will keep {split} for training, move {len(imgs) - split} to validation")

# Step 4: Move 15% to validation (both images AND labels)
count = 0
for f in imgs[split:]:
    lbl = os.path.splitext(f)[0] + ".txt"
    lbl_path = os.path.join(lbl_dir, lbl)
    
    if os.path.exists(lbl_path):
        shutil.move(os.path.join(img_dir, f), os.path.join(val_img, f))
        shutil.move(lbl_path, os.path.join(val_lbl, lbl))
        count += 1

print(f"\nDone! Moved {count} image+label pairs to valid/")
print(f"Train images: {len(os.listdir(img_dir))}")
print(f"Train labels: {len(os.listdir(lbl_dir))}")
print(f"Valid images: {len(os.listdir(val_img))}")
print(f"Valid labels: {len(os.listdir(val_lbl))}")
