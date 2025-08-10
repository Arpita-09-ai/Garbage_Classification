import os
import shutil
import random
from config import ORIGINAL_DATASET_PATH, SPLIT_DATASET_PATH, SPLIT_RATIOS

def split_data():
    original_path = ORIGINAL_DATASET_PATH
    output_base = SPLIT_DATASET_PATH

    # Safety check: don't allow deleting original dataset
    if os.path.abspath(original_path) == os.path.abspath(output_base):
        raise ValueError("Output path cannot be the same as the original dataset path!")

    # Create base output folder if it doesn't exist
    os.makedirs(output_base, exist_ok=True)

    # Clear only train/val/test inside output folder
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_base, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
            print(f"Cleared existing split folder: {split_dir}")

    # Create empty train/val/test folders
    for split in ['train', 'val', 'test']:
        for class_name in os.listdir(original_path):
            class_dir = os.path.join(original_path, class_name)
            if os.path.isdir(class_dir):
                os.makedirs(os.path.join(output_base, split, class_name), exist_ok=True)

    # Split and copy images
    for class_name in os.listdir(original_path):
        class_dir = os.path.join(original_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [img for img in os.listdir(class_dir) if not img.startswith('.')]
        random.shuffle(images)

        total = len(images)
        train_end = int(SPLIT_RATIOS[0] * total)
        val_end = int((SPLIT_RATIOS[0] + SPLIT_RATIOS[1]) * total)

        for i, img in enumerate(images):
            src = os.path.join(class_dir, img)
            if i < train_end:
                dst_folder = "train"
            elif i < val_end:
                dst_folder = "val"
            else:
                dst_folder = "test"

            dst = os.path.join(output_base, dst_folder, class_name, img)
            shutil.copy(src, dst)

    print(f"✅ Dataset successfully split into train/val/test at: {output_base}")

if __name__ == '__main__':
    split_data()
