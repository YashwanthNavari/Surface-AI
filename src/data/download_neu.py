"""
NEU Surface Defect Database Ingestion and Verification Script
Downloads the standard 6-class NEU surface defect dataset,
organizes into class directories, and runs thorough integrity checks.
"""
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image
from src.config import RAW_DATA_DIR, CLASSES

CLONE_URL = "https://github.com/Marfbin/NEU-DET-with-yolov8.git"
TEMP_DIR = RAW_DATA_DIR.parent / "_temp_neu_clone"

def download_and_organize():
    print(f"[1/4] Preparing target directory: {RAW_DATA_DIR}")
    for cls_name in CLASSES:
        (RAW_DATA_DIR / cls_name).mkdir(parents=True, exist_ok=True)

    # Check if already downloaded and verified
    counts = {cls_name: len(list((RAW_DATA_DIR / cls_name).glob("*.jpg"))) for cls_name in CLASSES}
    if all(count == 300 for count in counts.values()):
        print("NEU dataset already downloaded and complete (300 images/class, 1800 total).")
        return verify_integrity()

    print(f"[2/4] Cloning repository snapshot from {CLONE_URL}...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # Clone shallow depth to save bandwidth and time (~50MB)
    cmd = ["git", "clone", "--depth", "1", CLONE_URL, str(TEMP_DIR)]
    subprocess.run(cmd, check=True)

    print("[3/4] Extracting and organizing 1,800 images into defect categories...")
    # Find all jpg files in the cloned directory
    all_jpgs = list(TEMP_DIR.rglob("*.jpg"))
    copied_count = {cls_name: 0 for cls_name in CLASSES}

    for img_path in all_jpgs:
        filename = img_path.name.lower()
        target_class = None

        if filename.startswith("crazing"):
            target_class = "crazing"
        elif filename.startswith("inclusion"):
            target_class = "inclusion"
        elif filename.startswith("patches"):
            target_class = "patches"
        elif filename.startswith("pitted"):
            target_class = "pitted_surface"
        elif filename.startswith("rolled-in") or filename.startswith("rolled_in"):
            target_class = "rolled-in_scale"
        elif filename.startswith("scratches"):
            target_class = "scratches"

        if target_class:
            dest_path = RAW_DATA_DIR / target_class / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
            copied_count[target_class] += 1

    print("Copy summary:", copied_count)

    # Clean up temporary git clone
    print("Cleaning up temporary clone...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    return verify_integrity()

def verify_integrity():
    print("[4/4] Verifying dataset integrity and image formats...")
    total_valid = 0
    errors = []
    resolutions = set()

    for cls_name in CLASSES:
        cls_dir = RAW_DATA_DIR / cls_name
        files = list(cls_dir.glob("*.jpg"))
        print(f"  - Class '{cls_name}': {len(files)} files found")
        if len(files) != 300:
            errors.append(f"Class '{cls_name}' has {len(files)} files, expected 300.")

        for f in files:
            try:
                with Image.open(f) as img:
                    img.verify()
                # Re-open to read size and mode (verify closes image)
                with Image.open(f) as img:
                    resolutions.add(img.size)
                    total_valid += 1
            except Exception as e:
                errors.append(f"Corrupt image {f}: {e}")

    print(f"\nVerification Results:")
    print(f"Total valid images verified: {total_valid} / 1800")
    print(f"Distinct image resolutions found: {resolutions}")

    if errors:
        print("\nERRORS DETECTED:")
        for err in errors:
            print("  *", err)
        sys.exit(1)
    else:
        print("\nSUCCESS: NEU Surface Defect Database verified with 100% integrity!")

if __name__ == "__main__":
    download_and_organize()
