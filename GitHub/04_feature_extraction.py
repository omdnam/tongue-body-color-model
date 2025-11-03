import os
import cv2
import csv
import numpy as np
from skimage import color
from tqdm import tqdm
import glob

# 1. Configuration
PTC_CUTOFF = 17
VALUE_MIN = 45
VALUE_MAX = 55
IMAGE_DIRS = [
    './Analysis/images/05 train CC',
    './Analysis/images/07 test CC'
]
MASK_DIRS = [
    './Analysis/images/06 train mask',
    './Analysis/images/08 test mask'
]
OUTPUT_CSV = './Analysis/tongue_color_analysis_result.csv'

# 2. Helper Functions

def get_image_mask_paths(image_dirs, mask_dirs):

    paths = []
    for img_dir, mask_dir in zip(image_dirs, mask_dirs):
        for subfolder in os.listdir(img_dir):
            subfolder_path = os.path.join(img_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.png'):
                        img_path = os.path.join(subfolder_path, filename)
                        mask_path = os.path.join(mask_dir, subfolder, filename)
                        if os.path.exists(mask_path):
                            paths.append((img_path, mask_path, filename))
    return paths

def calculate_average_colors(image_bgr, mask):

    if mask is None or image_bgr is None:
        return ['NaN'] * 6

    mask_bin = cv2.inRange(mask, 1, 255)
    pixel_count = cv2.countNonZero(mask_bin)
    
    if pixel_count == 0:
        return ['NaN'] * 6

    mean_bgr = cv2.mean(image_bgr, mask=mask_bin)
    avg_b, avg_g, avg_r = mean_bgr[:3]

    avg_bgr_pixel = np.uint8([[[avg_b, avg_g, avg_r]]])
    avg_lab_pixel = cv2.cvtColor(avg_bgr_pixel, cv2.COLOR_BGR2LAB)
    avg_l, avg_a, avg_b_lab = avg_lab_pixel[0][0]

    return [
        round(float(avg_l), 3), round(float(avg_a), 3), round(float(avg_b_lab), 3),
        round(avg_r, 3), round(avg_g, 3), round(avg_b, 3)
    ]

# 3. Main Processing Block

def main():
    print("Starting feature extraction script")
    
    all_paths = get_image_mask_paths(IMAGE_DIRS, MASK_DIRS)
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Filename', 'tongue_Px_num', 'TB_Px_num', 'TC_Px_num', 'PTC',
            'Whole_L', 'Whole_a', 'Whole_b', 'Whole_R', 'Whole_G', 'Whole_B',
            'TB_L', 'TB_a', 'TB_b', 'TB_R', 'TB_G', 'TB_B',
            'TC_L', 'TC_a', 'TC_b', 'TC_R', 'TC_G', 'TC_B'
        ])

        for img_path, mask_path, filename in tqdm(all_paths, desc="Extracting Features"):
            try:
                tongue_bgr = cv2.imread(img_path)
                tongue_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if tongue_bgr is None or tongue_mask is None:
                    continue

                tongue_area = cv2.bitwise_and(tongue_bgr, tongue_bgr, mask=tongue_mask)

                tongue_area_rgb = cv2.cvtColor(tongue_area, cv2.COLOR_BGR2RGB)
                tongue_area_lab = color.rgb2lab(tongue_area_rgb)
                _, a_channel, _ = cv2.split(tongue_area_lab)
                
                _, tongue_body_mask_ptc = cv2.threshold(a_channel.astype(np.uint8), PTC_CUTOFF, 255, cv2.THRESH_BINARY)
                
                tongue_coat_mask_ptc = cv2.bitwise_and(tongue_mask, cv2.bitwise_not(tongue_body_mask_ptc))

                hsv_image = cv2.cvtColor(tongue_bgr, cv2.COLOR_BGR2HSV)
                _, _, v_channel = cv2.split(hsv_image)
                value_mask = cv2.inRange(v_channel, VALUE_MIN, VALUE_MAX)

                final_body_mask = cv2.bitwise_and(tongue_body_mask_ptc, value_mask)
                final_coat_mask = cv2.bitwise_and(tongue_coat_mask_ptc, value_mask)

                px_tongue = cv2.countNonZero(tongue_mask)
                px_tb = cv2.countNonZero(tongue_body_mask_ptc)
                px_tc = px_tongue - px_tb if px_tongue > px_tb else 0
                ptc = round((px_tc / px_tongue) * 100, 3) if px_tongue > 0 else 0

                colors_whole = calculate_average_colors(tongue_bgr, tongue_mask)
                colors_tb = calculate_average_colors(tongue_bgr, final_body_mask)
                colors_tc = calculate_average_colors(tongue_bgr, final_coat_mask)

                row = [filename, px_tongue, px_tb, px_tc, ptc] + colors_whole + colors_tb + colors_tc
                writer.writerow(row)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nFeature extraction complete. Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
