import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import f1_score, classification_report

# =========================
# CONFIG
# =========================
PROBS_DIR = "probs"
SPEC_DIR = "probs/specialists"

# =========================
# LOAD FILES
# =========================
val_paths  = sorted(glob(os.path.join(PROBS_DIR, "val_probs_*.npz")))
test_paths = sorted(glob(os.path.join(PROBS_DIR, "test_probs_*.npz")))

spec_val_paths  = sorted(glob(os.path.join(SPEC_DIR, "val_probs_*.npz")))
spec_test_paths = sorted(glob(os.path.join(SPEC_DIR, "test_probs_*.npz")))

print("General models:")
for p in val_paths:
    print("-", os.path.basename(p))

print("\nSpecialists:")
for p in spec_val_paths:
    print("-", os.path.basename(p))

# =========================
# LABELS
# =========================
val_df = pd.read_csv("val_split.csv")

global_classes = sorted(val_df["label"].unique())
n_classes = len(global_classes)

class_to_idx = {c: i for i, c in enumerate(global_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

y_val = val_df["label"].map(class_to_idx).values

# =========================
# ALIGN
# =========================
def align_probs(probs, model_classes):
    aligned = np.zeros((probs.shape[0], n_classes))
    for i, c in enumerate(model_classes):
        aligned[:, class_to_idx[c]] = probs[:, i]
    return aligned

# =========================
# LOAD GENERAL
# =========================
general_val, general_test = [], []

for vp, tp in zip(val_paths, test_paths):
    v = np.load(vp, allow_pickle=True)
    t = np.load(tp, allow_pickle=True)

    general_val.append(align_probs(v["probs"], v["classes"].tolist()))
    general_test.append(align_probs(t["probs"], t["classes"].tolist()))

# =========================
# LOAD SPECIALISTS
# =========================
def get_group(path):
    name = path.lower()
    if "myeloid" in name: return "myeloid"
    if "lymphoid" in name: return "lymphoid"
    if "neutrophil" in name: return "neutrophil"
    return None

spec_val = {"myeloid": [], "lymphoid": [], "neutrophil": []}
spec_test = {"myeloid": [], "lymphoid": [], "neutrophil": []}

for vp, tp in zip(spec_val_paths, spec_test_paths):
    v = np.load(vp, allow_pickle=True)
    t = np.load(tp, allow_pickle=True)

    group = get_group(vp)
    if group:
        spec_val[group].append(align_probs(v["probs"], v["classes"].tolist()))
        spec_test[group].append(align_probs(t["probs"], t["classes"].tolist()))

# =========================
# GROUPS
# =========================
groups = {
    "myeloid": ['MY', 'MMY', 'PMY'],
    "lymphoid": ['LY', 'VLY', 'PLY'],
    "neutrophil": ['BNE', 'SNE']
}

group_indices = {
    g: [class_to_idx[c] for c in cls]
    for g, cls in groups.items()
}

# =========================
# COMBINE CLASSWISE
# =========================
def combine_classwise(weights, probs_list):
    combined = np.zeros_like(probs_list[0])
    for m in range(len(probs_list)):
        combined += probs_list[m] * weights[m]
    return combined

# =========================
# APPLY SPECIALISTS (NEW)
# =========================
def apply_specialists(base, spec_dict, alpha_dict, sp_thresh_dict):
    combined = base.copy()

    for i in range(len(base)):
        pred = np.argmax(base[i])

        for g, idxs in group_indices.items():
            if len(spec_dict[g]) == 0:
                continue

            if pred not in idxs:
                continue

            sp = np.mean([m[i] for m in spec_dict[g]], axis=0)
            sp_group = sp[idxs]

            sp_conf = sp_group.max()
            if sp_conf < sp_thresh_dict[g]:
                continue

            sp_group = sp_group / (sp_group.sum() + 1e-8)
            alpha = alpha_dict[g]

            combined[i, idxs] = (
                alpha * sp_group +
                (1 - alpha) * combined[i, idxs]
            )

    return combined

# =========================
# INDIVIDUAL SCORES
# =========================
print("\n=== INDIVIDUAL MODELS ===")

for i, p in enumerate(general_val):
    preds = np.argmax(p, axis=1)
    print(f"General {i}: {f1_score(y_val, preds, average='macro'):.4f}")

# =========================
# FIND BEST GENERAL ENSEMBLE
# =========================
print("\n=== SEARCHING GENERAL ENSEMBLE ===")

n_models = len(general_val)
best_score = 0
best_weights = None

for i in range(2000):
    weights = np.zeros((n_models, n_classes))
    for c in range(n_classes):
        weights[:, c] = np.random.dirichlet(np.ones(n_models))

    combined = combine_classwise(weights, general_val)
    preds = np.argmax(combined, axis=1)
    score = f1_score(y_val, preds, average="macro")

    if score > best_score:
        best_score = score
        best_weights = weights

print("\n=== ENSEMBLE_BASE ===")
base_val = combine_classwise(best_weights, general_val)
base_preds = np.argmax(base_val, axis=1)
base_score = f1_score(y_val, base_preds, average="macro")
print("Macro F1:", base_score)

# =========================
# SAVE BASE SUBMISSION
# =========================
base_test = combine_classwise(best_weights, general_test)
test_preds = np.argmax(base_test, axis=1)

test_df = pd.read_csv("test_metadata.csv")

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "label": [idx_to_class[i] for i in test_preds]
})

os.makedirs("submissions", exist_ok=True)
submission.to_csv("submissions/ensemble_base.csv", index=False)

print("Saved → submissions/ensemble_base.csv")

# =========================
# OPTIMIZE SPECIALISTS
# =========================
print("\n=== OPTIMIZING SPECIALISTS ===")

best_spec_score = base_score
best_params = None
best_combined = base_val.copy()

groups_list = list(group_indices.keys())

for i in range(2000):

    alpha_dict = {g: np.random.uniform(0.0, 0.6) for g in groups_list}
    sp_thresh_dict = {g: np.random.uniform(0.4, 0.9) for g in groups_list}

    combined = apply_specialists(
        base_val,
        spec_val,
        alpha_dict,
        sp_thresh_dict
    )

    preds = np.argmax(combined, axis=1)
    score = f1_score(y_val, preds, average="macro")

    if score > best_spec_score:
        best_spec_score = score
        best_params = (alpha_dict.copy(), sp_thresh_dict.copy())
        best_combined = combined

    if i % 50 == 0:
        print(f"Iter {i} | Best F1: {best_spec_score:.4f}")

print("\nBest specialist F1:", best_spec_score)
print("Params:", best_params)

# =========================
# FINAL REPORT
# =========================
final_preds = np.argmax(best_combined, axis=1)

print("\n=== FINAL REPORT ===")
print(classification_report(
    y_val,
    final_preds,
    target_names=global_classes,
    digits=4
))

# =========================
# SAVE SPECIALIST SUBMISSION
# =========================
alpha_dict, sp_thresh_dict = best_params

combined_test = apply_specialists(
    base_test,
    spec_test,
    alpha_dict,
    sp_thresh_dict
)

test_preds = np.argmax(combined_test, axis=1)

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "label": [idx_to_class[i] for i in test_preds]
})

submission.to_csv("submissions/ensemble_specialists.csv", index=False)

print("Saved → submissions/ensemble_specialists.csv")