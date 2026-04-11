import argparse
import yaml
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

import wandb
from utils import WBCDataset


# -------------------------
# CONFIG
# -------------------------
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.lr:
        config["lr"] = args.lr
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.epochs:
        config["epochs"] = args.epochs

    return config


# -------------------------
# SCHEDULER
# -------------------------
def get_scheduler(optimizer, config):
    sched_cfg = config.get("scheduler", {})
    name = sched_cfg.get("name", None)

    if name is None:
        return None

    if name == "cosine_warmup":
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

        warmup_epochs = sched_cfg.get("warmup_epochs", 5)
        total_epochs = config["epochs"]

        warmup = LinearLR(
            optimizer,
            start_factor=sched_cfg.get("start_factor", 0.01),
            total_iters=warmup_epochs
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )

    else:
        raise ValueError(f"Unknown scheduler: {name}")


# -------------------------
# LOGGING
# -------------------------
def log_confusion_matrix(all_true, all_preds, class_names):
    cm = confusion_matrix(all_true, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    wandb.log({"confusion_matrix": wandb.Image(fig)})

    report = classification_report(all_true, all_preds, target_names=class_names, output_dict=True)
    wandb.log({
        "classification_report": wandb.Table(
            dataframe=pd.DataFrame(report).transpose()
        )
    })


# -------------------------
# MAIN
# -------------------------
def main():
    config = load_config()

    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(project=config["project"], config=config)

    device = torch.device(config["device"])

    # -------------------------
    # DATASETS
    # -------------------------
    train_dataset = WBCDataset(
        config["class_names"],
        config["train_csv"],
        config["train_dir"],
        augment=config["augmentation"],
        center_crop=config["center_crop"]
    )

    val_dataset = WBCDataset(
        config["class_names"],
        config["val_csv"],
        config["train_dir"],
        augment=False,
        center_crop=config["center_crop"]
    )

    # -------------------------
    # IMBALANCE
    # -------------------------
    train_df = pd.read_csv(config["train_csv"])
    labels = train_df["label"].values
    strategy = config.get("imbalance_strategy")

    classes = np.array(config["class_names"])

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    sampler = None
    if strategy in ["sampler", "both"]:
        class_count = train_df["label"].value_counts().to_dict()
        sample_weights = [1.0 / class_count[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    # -------------------------
    # DATALOADERS
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------
    # MODEL (ConvNeXt)
    # -------------------------
    model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(config["class_names"]))

    model = model.to(device)

    # -------------------------
    # LOSS
    # -------------------------
    if strategy == "class_weights":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif strategy == "sampler":
        criterion = nn.CrossEntropyLoss()
    elif strategy == "none":
        criterion = nn.CrossEntropyLoss()
    elif strategy == "both":
        criterion = nn.CrossEntropyLoss(weight=class_weights ** 0.5)
    else:
        raise ValueError(f"Invalid imbalance_strategy: {strategy}")
    # -------------------------
    # OPTIMIZER
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )

    scheduler = get_scheduler(optimizer, config)

    # -------------------------
    # SAVE
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{config['weights_path']}_{timestamp}.pth"

    best_val_f1 = 0.0

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(config["epochs"]):

        model.train()
        running_loss = 0

        for images, labels, _ in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        if scheduler:
            scheduler.step()

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        all_preds, all_true = [], []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.numpy())

        val_f1 = f1_score(all_true, all_preds, average="macro")

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1
            }, model_path)

            wandb.save(model_path)
            print("New best model saved!")

            log_confusion_matrix(all_true, all_preds, config["class_names"])

    print("Best Val F1:", best_val_f1)


if __name__ == "__main__":
    main()