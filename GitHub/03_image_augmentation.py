import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
import glob

# 1. Configuration
RANDOM_SEED = 90
AUGMENTATIONS_PER_IMAGE = 5
IMAGE_EXTENSIONS = ['.png']
TRAIN_IMG_DIR = './Analysis/images/05 train CC'
TRAIN_MASK_DIR = './Analysis/images/06 train mask'

# 2. Augmentation Definitions
def get_geometric_transform():

    return A.Compose([
        A.Rotate(limit=5, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=1.0),
        A.HorizontalFlip(p=0.2),
    ])

def get_brightness_transform():

    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit=0.0, p=1.0)
    ])

# 3. Core Functions

def get_image_mask_pairs(image_root, mask_root):

    pairs = []
    for img_path in glob.glob(os.path.join(image_root, '**', '*'), recursive=True):
        if os.path.splitext(img_path)[1].lower() in IMAGE_EXTENSIONS:
            rel_path = os.path.relpath(img_path, image_root)
            mask_path = os.path.join(mask_root, rel_path)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
    return pairs

def augment_and_save(image_path, mask_path, count, geo_transform, bright_transform):

    try:
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Warning: Could not read image or mask. Skipping {image_path}")
            return

        for i in range(count):
            augmented = geo_transform(image=img, mask=mask)
            aug_img, aug_mask = augmented['image'], augmented['mask']

            aug_img = bright_transform(image=aug_img)['image']

            base_name, ext = os.path.splitext(os.path.basename(image_path))
            aug_filename = f"{base_name}_aug{i+1}{ext}"

            aug_img_path = os.path.join(os.path.dirname(image_path), aug_filename)
            cv2.imwrite(aug_img_path, aug_img)
            
            aug_mask_path = os.path.join(os.path.dirname(mask_path), aug_filename)
            cv2.imwrite(aug_mask_path, aug_mask)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 4. Main Execution Block

def main():

    print("Starting image augmentation script")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    geo_transform = get_geometric_transform()
    bright_transform = get_brightness_transform()

    pairs = get_image_mask_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    if not pairs:
        print("No image-mask pairs found. Please check the input directories.")
        return
    
    print(f"Found {len(pairs)} image-mask pairs to augment.")

    for img_path, mask_path in tqdm(pairs, desc="Augmenting images"):
        augment_and_save(img_path, mask_path, AUGMENTATIONS_PER_IMAGE, geo_transform, bright_transform)

    print("\nImage augmentation complete!")

if __name__ == '__main__':
    main()