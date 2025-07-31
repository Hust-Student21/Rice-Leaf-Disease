import os
import random
import shutil
from pathlib import Path

# === Configuration ===
SOURCE_DIR = r"c:\Users\ADMIN\OneDrive - Hanoi University of Science and Technology\Desktop\drive\three_disease\images\train"  # Your training image folder
TARGET_DIR = "calib"                  # Where calibration images go
N_CALIB_IMAGES = 50                   # How many images to copy
EXTENSIONS = [".jpg", ".jpeg", ".png"]
# =======================

def prepare_calibration_images(source_dir, target_dir, n_images):
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(exist_ok=True)

    all_images = [p for p in source.glob("*") if p.suffix.lower() in EXTENSIONS]

    if len(all_images) < n_images:
        raise ValueError(f"Not enough images in {source_dir} (found {len(all_images)}, need {n_images})")

    sampled = random.sample(all_images, n_images)

    print(f"Copying {n_images} images from {source_dir} to {target_dir}...")

    for img_path in sampled:
        shutil.copy(img_path, target / img_path.name)

    print("Done! Calibration images are ready.")

if __name__ == "__main__":
    prepare_calibration_images(SOURCE_DIR, TARGET_DIR, N_CALIB_IMAGES)
