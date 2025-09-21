import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================== CONFIG ==================
root_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-4"
GT_DIRS = [
    os.path.join(root_dir, "AnnotationMasks"),
    # os.path.join(root_dir, "AnnotationMasks_1"),
    os.path.join(root_dir, "ACCEPTED_MASKS"),
    os.path.join(root_dir, "REWORK_MASKS"),
]

PRED_DIR = os.path.join(root_dir, "process_distress_results")
SAVE_CSV = os.path.join(root_dir, "AMRAWATI-TALEGAON_s4_-mask_metrics.csv")

# --- Color map (RGB) → (ID, Name) ---
COLOR_MAP = {
    (0, 0, 0): (0, "Background"),
    (255, 0, 0): (1, "Alligator"),
    (0, 0, 255): (2, "Transverse Crack"),
    (0, 255, 0): (3, "Longitudinal Crack"),
    (139, 69, 19): (4, "Pothole"),
    (255, 165, 0): (5, "Patches"),
    (255, 0, 255): (6, "Multiple Crack"),
    (0, 255, 255): (7, "Spalling"),
    (0, 128, 0): (8, "Corner Break"),
    (255, 100, 203): (9, "Sealed Joint Transverse"),
    (199, 21, 133): (10, "Sealed Joint Longitudinal"),
    (128, 0, 128): (11, "Punchout"),
    (112, 102, 255): (12, "Popout"),
    (255, 255, 255): (13, "Unclassified"),
    (255, 215, 0): (14, "Cracking"),
}

IGNORE_CLASSES = {"Patches", "Spalling", "Corner Break",
                  "Sealed Joint Transverse", "Sealed Joint Longitudinal",
                  "Punchout", "Popout", "Cracking"}

# ================== HELPERS ==================
# Build lookup for fast mapping (RGB24 → index)
lut = np.zeros((256, 256, 256), dtype=np.uint8)
for rgb, (idx, _) in COLOR_MAP.items():
    lut[rgb] = idx


def color_to_index(mask):
    """Fast RGB mask → index mask using lookup"""
    return lut[mask[..., 0], mask[..., 1], mask[..., 2]]


def compute_classwise_overlap(gt, pred):
    stats = []
    classes = np.union1d(np.unique(gt), np.unique(pred))

    for c in classes:
        if c == 0:
            continue
        class_name = [n for _, (i, n) in COLOR_MAP.items() if i == c][0]
        if class_name in IGNORE_CLASSES:
            continue

        gt_mask = (gt == c)
        pred_mask = (pred == c)
        if not gt_mask.any() and not pred_mask.any():
            continue

        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        gt_count, pred_count = gt_mask.sum(), pred_mask.sum()

        stats.append({
            "class_name": class_name,
            "GT_pixels": int(gt_count),
            "Pred_pixels": int(pred_count),
            "Intersection": int(inter),
            "IoU": inter / (union + 1e-7),
            "Dice": (2 * inter) / (gt_count + pred_count + 1e-7)
        })
    return stats


def compute_detection_metrics(gt, pred, iou_thresh=0.3):
    stats = []
    classes = np.union1d(np.unique(gt), np.unique(pred))

    for c in classes:
        if c == 0:
            continue
        class_name = [n for _, (i, n) in COLOR_MAP.items() if i == c][0]
        if class_name in IGNORE_CLASSES:
            continue

        # Connected components
        gt_labels = cv2.connectedComponents((gt == c).astype(np.uint8), 8)[1]
        pred_labels = cv2.connectedComponents((pred == c).astype(np.uint8), 8)[1]

        gt_ids = np.unique(gt_labels)[1:]  # skip background
        pred_ids = np.unique(pred_labels)[1:]

        matched_gt, matched_pred = set(), set()
        for gi in gt_ids:
            g = (gt_labels == gi)
            for pi in pred_ids:
                p = (pred_labels == pi)
                inter = np.logical_and(g, p).sum()
                union = g.sum() + p.sum() - inter
                if union > 0 and inter / union >= iou_thresh:
                    matched_gt.add(gi)
                    matched_pred.add(pi)

        TP = len(matched_gt)
        FN = len(gt_ids) - TP
        FP = len(pred_ids) - TP

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        stats.append({
            "class_name": class_name,
            "GT_objects": len(gt_ids),
            "Pred_objects": len(pred_ids),
            "TP": TP, "FP": FP, "FN": FN,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })
    return stats


# ================== MAIN ==================
# Collect GT files
gt_files = {}
for gt_dir in GT_DIRS:
    for f in os.listdir(gt_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            gt_files[f] = os.path.join(gt_dir, f)

pred_files = {f: os.path.join(PRED_DIR, f) for f in os.listdir(PRED_DIR)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))}

all_files = sorted(set(gt_files) | set(pred_files))

all_results = []
for fname in tqdm(all_files):
    gt_path, pred_path = gt_files.get(fname), pred_files.get(fname)

    if gt_path:
        gt_mask = color_to_index(cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB))
    else:
        gt_mask = np.zeros((1024, 419), np.uint8)  # fallback

    if pred_path:
        pred_mask = color_to_index(cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB))
    else:
        pred_mask = np.zeros_like(gt_mask)

    seg_stats = compute_classwise_overlap(gt_mask, pred_mask)
    det_stats = compute_detection_metrics(gt_mask, pred_mask, iou_thresh=0.3)

    for cname in {s["class_name"] for s in seg_stats} | {d["class_name"] for d in det_stats}:
        seg = next((s for s in seg_stats if s["class_name"] == cname), {})
        det = next((d for d in det_stats if d["class_name"] == cname), {})
        all_results.append({
            "filename": fname,
            "status": "ok" if (gt_path and pred_path) else ("missing_pred" if gt_path else "missing_gt"),
            "class_name": cname,
             **{k: seg.get(k, 0) for k in ["GT_pixels", "Pred_pixels", "Intersection", "IoU", "Dice"]},
            **{k: det.get(k, 0) for k in ["GT_objects", "Pred_objects", "TP", "FP", "FN", "Precision", "Recall", "F1"]}
        })

df = pd.DataFrame(all_results)
pd.DataFrame(df).to_csv(SAVE_CSV, index=False)


# ================= CLASS-WISE METRICS =================
classwise_results = []
for cname, g in df.groupby("class_name"):
    if cname in ["ALL"]:  # skip summary rows if rerun
        continue

    mean_iou = g["IoU"].mean()
    mean_dice = g["Dice"].mean()

    TP = g["TP"].sum()
    FP = g["FP"].sum()
    FN = g["FN"].sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    acc = g["Intersection"].sum() / g["GT_pixels"].sum() if g["GT_pixels"].sum() > 0 else 0.0

    classwise_results.append({
        "filename": "GLOBAL",
        "class_name": cname,
        "status": "class_summary",

        "GT_pixels": g["GT_pixels"].sum(),
        "Pred_pixels": g["Pred_pixels"].sum(),
        "Intersection": g["Intersection"].sum(),
        "IoU": mean_iou,
        "Dice": mean_dice,

        "GT_objects": g["GT_objects"].sum(),
        "Pred_objects": g["Pred_objects"].sum(),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": acc
    })

# ================= OVERALL METRICS =================
mean_iou = df["IoU"].mean()
mean_dice = df["Dice"].mean()

TP = df["TP"].sum()
FP = df["FP"].sum()
FN = df["FN"].sum()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

acc = df["Intersection"].sum() / df["GT_pixels"].sum() if df["GT_pixels"].sum() > 0 else 0.0

summary = {
    "filename": "GLOBAL",
    "class_name": "ALL",
    "status": "summary",

    "GT_pixels": df["GT_pixels"].sum(),
    "Pred_pixels": df["Pred_pixels"].sum(),
    "Intersection": df["Intersection"].sum(),
    "IoU": mean_iou,
    "Dice": mean_dice,

    "GT_objects": df["GT_objects"].sum(),
    "Pred_objects": df["Pred_objects"].sum(),
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Accuracy": acc
}

# ================= FINAL CONCAT =================
df = pd.concat([df, pd.DataFrame(classwise_results + [summary])], ignore_index=True)

# Save CSV
df.to_csv(SAVE_CSV, index=False)
print(f"✅ Results saved to {SAVE_CSV}")

