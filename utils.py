from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
import os
import pandas as pd
import config

class WBCDataset(Dataset):
    def __init__(self, df_path, img_dir, augment=False):
        self.df = pd.read_csv(df_path).reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.class_to_idx = config.CLASS_TO_IDX
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
        
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        image = (image - self.mean) / self.std

        label_idx = self.class_to_idx[label]
        return image, torch.tensor(label_idx, dtype=torch.long), img_name

class WBCTestDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        image = (image - self.mean) / self.std

        return image
    
def is_noisy_v2(img, threshold=10):
    median = cv2.medianBlur(img, 3)
    diff = np.abs(img.astype(np.int16) - median.astype(np.int16))
    score = np.mean(diff)
    return score > threshold