import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


# --- Noise score ---
def noise_score(img):
    median = cv2.medianBlur(img, 3)
    diff = np.abs(img.astype(np.int16) - median.astype(np.int16))
    return np.mean(diff)


# --- Classification ---
def classify_noise(score, t1=20, t2=40):
    if score < t1:
        return "clean"
    elif score < t2:
        return "noisy"
    else:
        return "very_noisy"


# --- Adaptive denoising ---
def adaptive_denoise(img, h_noisy=20, h_very_noisy=40):
    score = noise_score(img)
    category = classify_noise(score)

    if category == "clean":
        return img
    
    elif category == "noisy":
        return cv2.fastNlMeansDenoisingColored(img, None, h_noisy, h_noisy, 7, 21)
    
    else:
        return cv2.fastNlMeansDenoisingColored(img, None, h_very_noisy, h_very_noisy, 7, 21)


# --- Main ---
def process_folder(input_folder, h_noisy=20, h_very_noisy=40):
    output_folder = input_folder.rstrip("/\\") + "_denoised"
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)

    for filename in tqdm(files):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path)
        if img is None:
            continue

        denoised = adaptive_denoise(img, h_noisy, h_very_noisy)

        cv2.imwrite(output_path, denoised)

    print(f"\nDenoised images saved to: {output_folder}")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise dataset with adaptive NLM")

    parser.add_argument("data_folder", type=str, help="Path to image folder")
    parser.add_argument("--h_noisy", type=int, default=20, help="Denoising strength for noisy images")
    parser.add_argument("--h_very_noisy", type=int, default=40, help="Denoising strength for very noisy images")

    args = parser.parse_args()

    process_folder(args.data_folder, args.h_noisy, args.h_very_noisy)