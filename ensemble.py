import os
import numpy as np
import pandas as pd

from glob import glob
from sklearn.metrics import f1_score, classification_report

# =========================
# LOAD FILES
# =========================
PROBS_DIR = "probs"

val_paths  = sorted(glob(os.path.join(PROBS_DIR, "val_probs_*.npy")))
test_paths = sorted(glob(os.path.join(PROBS_DIR, "test_probs_*.npy")))

assert len(val_paths) == len(test_paths), "Mismatch val/test files"

print("Models found:")
for p in val_paths:
    print("-", os.path.basename(p))

val_probs  = [np.load(p) for p in val_paths]
test_probs = [np.load(p) for p in test_paths]

# =========================
# LABELS
# =========================
val_df = pd.read_csv("val_split.csv")

classes = sorted(val_df["label"].unique())
y_val = val_df["label"].astype("category").cat.codes.values

idx_to_class = {i: cls for i, cls in enumerate(classes)}

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(weights, probs_list, y_true):
    combined = sum(w * p for w, p in zip(weights, probs_list))
    preds = np.argmax(combined, axis=1)
    score = f1_score(y_true, preds, average="macro")
    return score, preds

# =========================
# BASELINE: EACH MODEL
# =========================
print("\n=== Individual model scores ===")

for i, p in enumerate(val_probs):
    preds = np.argmax(p, axis=1)
    score = f1_score(y_val, preds, average="macro")
    print(f"Model {i}: {score:.4f}")

# =========================
# RANDOM SEARCH FOR WEIGHTS
# =========================
print("\n=== Searching best weights ===")

n_models = len(val_probs)
best_score = 0
best_weights = None

for i in range(500):  # increase if needed
    weights = np.random.dirichlet(np.ones(n_models))

    score, _ = evaluate(weights, val_probs, y_val)

    if score > best_score:
        best_score = score
        best_weights = weights

    if i % 50 == 0:
        print(f"Iter {i} | Best F1: {best_score:.4f}")

print("\nBest macro F1:", best_score)
print("Best weights:", best_weights)

# =========================
# FINAL VALIDATION REPORT
# =========================
final_score, val_preds = evaluate(best_weights, val_probs, y_val)

print("\n=== FINAL VALIDATION ===")
print("Macro F1:", final_score)

report = classification_report(
    y_val,
    val_preds,
    target_names=classes,
    digits=4
)

print("\nPer-class F1:")
print(report)

# =========================
# APPLY TO TEST
# =========================
combined_test = sum(w * p for w, p in zip(best_weights, test_probs))
test_preds = np.argmax(combined_test, axis=1)

labels_str = [idx_to_class[i] for i in test_preds]

# =========================
# SAVE SUBMISSION
# =========================
test_df = pd.read_csv("test_metadata.csv")

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "label": labels_str
})

os.makedirs("submissions", exist_ok=True)
submission.to_csv("submissions/ensemble_submission.csv", index=False)

print("\nSubmission saved → submissions/ensemble_submission.csv")