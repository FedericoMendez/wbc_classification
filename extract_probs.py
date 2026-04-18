import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torchvision import models
from torch.utils.data import DataLoader

from utils import WBCTestDataset, WBCDataset

# =========================
# ARGPARSE
# =========================
parser = argparse.ArgumentParser(description="Extract probabilities for ensembling")

parser.add_argument("--weights", type=str, required=True)
parser.add_argument("--model", type=str, required=True, choices=["resnet50", "convnext_base"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--center_crop", action="store_true")
parser.add_argument("--denoise", action="store_true")

args = parser.parse_args()

WEIGHTS_PATH = args.weights
MODEL_NAME = args.model
BATCH_SIZE = args.batch_size
CENTER_CROP = args.center_crop
DENOISE = args.denoise

print(f"Model: {MODEL_NAME}")
print(f"Weights: {WEIGHTS_PATH}")
print(f"Center crop: {CENTER_CROP}")
print(f"Denoise: {DENOISE}")

# =========================
# SAFETY
# =========================
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL BUILDING
# =========================
def build_model(name):
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 13)

    elif name == "convnext_base":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 13)

    return model

# =========================
# LOAD MODEL
# =========================
model = build_model(MODEL_NAME)

ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# =========================
# DATA LOADER
# =========================
def make_loader(csv_path, img_dir, test=False):
    if test:
        dataset = WBCTestDataset(
            csv_path,
            img_dir,
            denoise=DENOISE,
            center_crop=CENTER_CROP
        )
    else:
        dataset = WBCDataset(
            class_names=['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY'],
            df_path=csv_path,
            img_dir=img_dir,
            augment=False,
            center_crop=CENTER_CROP
        )

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

val_loader  = make_loader("val_split.csv", "data/train_denoised")
test_loader = make_loader("test_metadata.csv", "data/test_denoised", test=True)

# =========================
# PROBABILITY EXTRACTION
# =========================
def extract_probs(model, dataloader, device):
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)

            outputs = model(images)          # logits
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())

    return torch.cat(all_probs).numpy()

print("\nExtracting probabilities...")

val_probs  = extract_probs(model, val_loader, DEVICE)
test_probs = extract_probs(model, test_loader, DEVICE)

# =========================
# SANITY CHECK
# =========================
print("Val probs shape:", val_probs.shape)
print("Example row sum (should be 1):", val_probs[0].sum())

# =========================
# SAVE
# =========================
base_name = os.path.splitext(os.path.basename(WEIGHTS_PATH))[0]
suffix = f"{MODEL_NAME}_crop{int(CENTER_CROP)}_denoise{int(DENOISE)}"

os.makedirs("probs", exist_ok=True)

val_path  = f"probs/val_probs_{base_name}_{suffix}.npy"
test_path = f"probs/test_probs_{base_name}_{suffix}.npy"

np.save(val_path, val_probs)
np.save(test_path, test_probs)

print(f"\nSaved:")
print(val_path)
print(test_path)

print("\nDone.")