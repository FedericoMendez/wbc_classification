from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
import os
import pandas as pd

class WBCDataset(Dataset):
    def __init__(self, class_names, df_path, img_dir, augment=False, center_crop=True):
        self.df = pd.read_csv(df_path).reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.denoise = denoise
        self.center_crop = center_crop
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['ID']
        label = self.df.iloc[idx]['label']

        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found or unreadable: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augment:

            # Random horizontal flip
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)

            # Random vertical flip
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 0)

            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()

            # Color jitter (brightness/contrast)
            alpha = 1.0 + np.random.uniform(-0.15, 0.15)  # contrast
            beta  = np.random.uniform(-0.05, 0.05)        # brightness
            image = np.clip(alpha * image + beta * 255, 0, 255).astype(np.uint8)
        
        if self.center_crop:
            image = apply_center_crop(image)
        else:
            image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        image = (image - self.mean) / self.std

        label_idx = self.class_to_idx[label]
        return image, torch.tensor(label_idx, dtype=torch.long), img_name

class WBCTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, denoise=True, center_crop = False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.denoise = denoise
        self.center_crop = center_crop
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.denoise:
            image = denoise(image)  
        
        if self.center_crop:
            image = apply_center_crop(image)
        else:
            image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        image = (image - self.mean) / self.std

        return image
    
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
    
# --- Denoising per category ---
def denoise(img):
    score = noise_score(img)
    category = classify_noise(score)
    if category == "clean":
        return img  # no change
    
    elif category == "noisy":
        # mild denoising
        return cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)
    
    else:  # very_noisy
        # stronger denoising
        return cv2.fastNlMeansDenoisingColored(img, None, 40, 40, 7, 21)
    
def apply_center_crop(image, size=224):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    half = size // 2
    return image[cy-half:cy+half, cx-half:cx+half]