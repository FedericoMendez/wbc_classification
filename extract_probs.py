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
parser.add_argument("--metadata", type=str, default="val_split.csv")
parser.add_argument("--model", type=str, required=True, choices=["resnet50", "convnext_base"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--center_crop", action="store_true")
parser.add_argument("--denoise", action="store_true")

parser.add_argument(
    "--classes",
    nargs="+",
    default=['BA','BL','BNE','EO','LY','MMY','MO','MY','PC','PLY','PMY','SNE','VLY']
)

args = parser.parse_args()

WEIGHTS_PATH = args.weights
MODEL_NAME = args.model
BATCH_SIZE = args.batch_size
CENTER_CROP = args.center_crop
DENOISE = args.denoise
CLASSES = args.classes
METADATA = args.metadata

print(f"Model: {MODEL_NAME}")
print(f"Class Names: {CLASSES}")
print(f"Metadata: {METADATA}")
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
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    elif name == "convnext_base":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))

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
            center_crop=CENTER_CROP
        )
    else:
        dataset = WBCDataset(
            class_names= CLASSES,
            df_path=csv_path,
            img_dir=img_dir,
            augment=False,
            center_crop=CENTER_CROP,
            allow_unknown_labels=True
        )

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
extra = ""
if DENOISE:
    extra= "_denoised"
val_loader  = make_loader(METADATA, f"data/train{extra}")
test_loader = make_loader("test_metadata.csv", f"data/test{extra}", test=True)

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

val_path  = f"probs/val_probs_{base_name}_{suffix}.npz"
test_path = f"probs/test_probs_{base_name}_{suffix}.npz"

np.savez(
    val_path,
    probs=val_probs,
    classes=np.array(CLASSES)
)

np.savez(
    test_path,
    probs=test_probs,
    classes=np.array(CLASSES)
)

print(f"\nSaved:")
print(val_path)
print(test_path)

print("\nDone.")