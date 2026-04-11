import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "resnet_wbc_finetuned_split.pth"
BATCH_SIZE = 128
EPOCHS = 40
CLASS_NAMES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
WANDKEY="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"
TRAIN_IMG_DIR="train"