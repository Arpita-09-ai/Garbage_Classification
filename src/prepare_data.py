import os
import shutil
import random
from config import ORIGINAL_DATASET_PATH, SPLIT_DATASET_PATH, SPLIT_RATIOS

def split_data():
    
    # Use paths from the config file
    original_path = ORIGINAL_DATASET_PATH
    output_base = SPLIT_DATASET_PATH

    # Clear existing dataset folders if they exist
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
        print(f"Removed existing directory: {output_base}")

    # Create output directories (train, val, test) for each class
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

    # print(f"Dataset successfully splitted into train/val/test at: {output_base}")

if __name__ == '__main__':
    split_data()