import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import f1_score, classification_report

# =========================
# LOAD PROBS
# =========================
PROBS_DIR = "probs"

val_paths  = sorted(glob(os.path.join(PROBS_DIR, "val_probs_*.npy")))
test_paths = sorted(glob(os.path.join(PROBS_DIR, "test_probs_*.npy")))

val_probs  = [np.load(p) for p in val_paths]
test_probs = [np.load(p) for p in test_paths]

n_models = len(val_probs)
n_classes = val_probs[0].shape[1]

print("Models:", len(val_probs))
print("Classes:", n_classes)

# =========================
# LABELS
# =========================
val_df = pd.read_csv("val_split.csv")

classes = sorted(val_df["label"].unique())
y_val = val_df["label"].astype("category").cat.codes.values

idx_to_class = {i: cls for i, cls in enumerate(classes)}

# =========================
# CLASS-WISE COMBINATION
# =========================
def combine_classwise(weights, probs_list):
    # weights shape: (n_models, n_classes)
    combined = np.zeros_like(probs_list[0])

    for m in range(n_models):
        combined += probs_list[m] * weights[m]

    return combined

def evaluate(weights):
    combined = combine_classwise(weights, val_probs)
    preds = np.argmax(combined, axis=1)
    score = f1_score(y_val, preds, average="macro")
    return score, preds

# =========================
# RANDOM SEARCH
# =========================
print("\n=== Class-wise search ===")

best_score = 0
best_weights = None

for i in range(2000):

    # sample weights per class
    weights = np.zeros((n_models, n_classes))

    for c in range(n_classes):
        w = np.random.dirichlet(np.ones(n_models))
        weights[:, c] = w

    score, _ = evaluate(weights)

    if score > best_score:
        best_score = score
        best_weights = weights

    if i % 200 == 0:
        print(f"Iter {i} | Best F1: {best_score:.4f}")

print("\nBest macro F1:", best_score)

# =========================
# FINAL REPORT
# =========================
final_score, val_preds = evaluate(best_weights)

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
combined_test = combine_classwise(best_weights, test_probs)
test_preds = np.argmax(combined_test, axis=1)

labels_str = [idx_to_class[i] for i in test_preds]

test_df = pd.read_csv("test_metadata.csv")

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "label": labels_str
})

os.makedirs("submissions", exist_ok=True)
submission.to_csv("submissions/ensemble_classwise.csv", index=False)

print("\nSaved → submissions/ensemble_classwise.csv")