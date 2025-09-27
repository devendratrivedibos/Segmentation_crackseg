import os
import pdb

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================== CONFIG ==================
root_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1"
GT_DIRS = [os.path.join(root_dir, "ACCEPTED_MASKS"), os.path.join(root_dir, "ACCEPTED_MASKS_1"), os.path.join(root_dir, "ACCEPTED_MASKS_2"), ]
PRED_DIR = os.path.join(root_dir, "process_distress_results")
SAVE_CSV = os.path.join(root_dir, "AMRAWATI-TALEGAON_1000.csv")

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

IGNORE_CLASSES = {
    "Spalling", "Corner Break",
    "Sealed Joint Transverse", "Sealed Joint Longitudinal",
    "Punchout", "Popout", "Cracking"
}

FILTER_PIXELS = {
    # "Alligator": 0,
    "Pothole": 1000,
    # "Patches": 0
}

# ================== HELPERS ==================
# LUT for fast mapping (RGB → index)
lut = np.zeros((256, 256, 256), dtype=np.uint8)
for rgb, (idx, _) in COLOR_MAP.items():
    lut[rgb] = idx

def color_to_index(mask):
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
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        gt_count, pred_count = gt_mask.sum(), pred_mask.sum()

        stats.append({
            "class_name": class_name,
            "Ground Truth Pixels": int(gt_count),
            "Prediction Pixels": int(pred_count),
            "Pred/Ground Ratio": pred_count / gt_count if gt_count > 0 else 0,
            "Intersection": int(inter),
            "IoU": inter / (union + 1e-7) if union > 0 else 0.0,
            "Dice": (2 * inter) / (gt_count + pred_count + 1e-7) if (gt_count + pred_count) > 0 else 0.0
        })
    return stats


def get_objects(mask, class_id, min_pixels=50):
    """
    Returns list of binary masks of connected components of given class_id
    with area >= min_pixels.
    """
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    objects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_pixels:
            obj_mask = (labels == i).astype(np.uint8)
            objects.append(obj_mask)
    return objects

def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + 1e-7) if union > 0 else 0.0

def compute_detection_metrics(gt_mask, pred_mask, color_map, iou_thresh=0.5, min_pixels=50, ignore_background=True):
    stats = []
    classes_in_use = np.unique(np.concatenate([np.unique(gt_mask), np.unique(pred_mask)]))
    for c in classes_in_use:
        if ignore_background and c == 0:
            continue
        class_name = [name for _, (idx, name) in color_map.items() if idx == c][0]
        if class_name in IGNORE_CLASSES:
            continue

        gt_objects_org = get_objects(gt_mask, c, min_pixels=0)
        pred_objects_org = get_objects(pred_mask, c, min_pixels=0)
        # --- Filtered objects ---
        class_min_pixels = FILTER_PIXELS.get(class_name, None)  # None means no filtering
        if class_min_pixels is not None:
            gt_objects_filt = get_objects(gt_mask, c, min_pixels=class_min_pixels)
            pred_objects_filt = get_objects(pred_mask, c, min_pixels=class_min_pixels)

        else:
            # If no filter specified, filtered = original
            gt_objects_filt = gt_objects_org
            pred_objects_filt = pred_objects_org

        matched_gt = set()
        matched_pred = set()
        for gi, g in enumerate(gt_objects_filt):
            for pi, p in enumerate(pred_objects_filt):
                if compute_iou(g, p) >= iou_thresh:
                    matched_gt.add(gi)
                    matched_pred.add(pi)

        TP = len(matched_gt)
        FN = len(gt_objects_filt) - TP
        FP = len(pred_objects_filt) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        stats.append({
            "class_name": class_name,
            "GT_objects_org": len(gt_objects_org),
            "Pred_objects_org": len(pred_objects_org),
            "GT_objects_filt": len(gt_objects_filt),
            "Pred_objects_filt": len(pred_objects_filt),
            "TP": TP, "FP": FP, "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })
    return stats



# ================== MAIN ==================
# Collect GT and Pred files
gt_files = {}
for gt_dir in GT_DIRS:
    for f in os.listdir(gt_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            gt_files[f] = os.path.join(gt_dir, f)
pred_files = {f: os.path.join(PRED_DIR, f) for f in os.listdir(PRED_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
all_files = sorted(set(gt_files) | set(pred_files))

all_results = []
for fname in tqdm(all_files):
    gt_path, pred_path = gt_files.get(fname), pred_files.get(fname)

    if gt_path:
        gt_mask = color_to_index(cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB))
    else:
        gt_mask = np.zeros((1024, 419), np.uint8)

    if pred_path:
        pred_mask = color_to_index(cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB))
    else:
        pred_mask = np.zeros_like(gt_mask)

    seg_stats = compute_classwise_overlap(gt_mask, pred_mask)
    det_stats = compute_detection_metrics(gt_mask, pred_mask, COLOR_MAP, iou_thresh=0.5, min_pixels=50)

    for cname in {s["class_name"] for s in seg_stats} | {d["class_name"] for d in det_stats}:
        seg = next((s for s in seg_stats if s["class_name"] == cname), {})
        det = next((d for d in det_stats if d["class_name"] == cname), {})
        all_results.append({
            "filename": fname,
            "class_name": cname,
            "status": "ok" if (gt_path and pred_path) else ("missing_pred" if gt_path else "missing_gt"),
            "Ground Truth Pixels": seg.get("Ground Truth Pixels", 0),
            "Prediction Pixels": seg.get("Prediction Pixels", 0),
            "Intersection": seg.get("Intersection", 0),
            "IoU": seg.get("IoU", 0.0),
            "Dice": seg.get("Dice", 0.0),
            "GT_objects_org": det.get("GT_objects_org", 0),
            "Pred_objects_org": det.get("Pred_objects_org", 0),
            "GT_objects_filt": det.get("GT_objects_filt", 0),
            "Pred_objects_filt": det.get("Pred_objects_filt", 0),
            "TP": det.get("TP", 0),
            "FP": det.get("FP", 0),
            "FN": det.get("FN", 0),
            "Precision": det.get("Precision", 0.0),
            "Recall": det.get("Recall", 0.0),
            "F1": det.get("F1", 0.0),
        })

df = pd.DataFrame(all_results)

# ================= CLASS-WISE SUMMARY =================
classwise_results = []
for cname, g in df.groupby("class_name"):
    mean_iou = g["IoU"].mean()
    mean_dice = g["Dice"].mean()
    TP, FP, FN = g["TP"].sum(), g["FP"].sum(), g["FN"].sum()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = g["Intersection"].sum() / g["Ground Truth Pixels"].sum() if g["Ground Truth Pixels"].sum() > 0 else 0
    classwise_results.append({
        "filename": "GLOBAL",
        "class_name": cname,
        "status": "class_summary",
        "Ground Truth Pixels": g["Ground Truth Pixels"].sum(),
        "Prediction Pixels": g["Prediction Pixels"].sum(),
        "Pred/Ground Ratio": g["Prediction Pixels"].sum() / g["Ground Truth Pixels"].sum() if g["Ground Truth Pixels"].sum() > 0 else 0,
        "Intersection": g["Intersection"].sum(),
        "IoU": mean_iou,
        "Dice": mean_dice,
        "GT_objects_org": g["GT_objects_org"].sum(),
        "GT_objects_filt": g["GT_objects_filt"].sum(),
        "Pred_objects_org": g["Pred_objects_org"].sum(),
        "Pred_objects_filt": g["Pred_objects_filt"].sum(),
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision, "Recall": recall, "F1": f1,
        "Accuracy": acc
    })

# ================= OVERALL SUMMARY =================
overall = df.sum(numeric_only=True)
overall_TP, overall_FP, overall_FN = overall["TP"], overall["FP"], overall["FN"]
overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0
overall_recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
overall_acc = overall["Intersection"] / overall["Ground Truth Pixels"] if overall["Ground Truth Pixels"] > 0 else 0

summary = {
    "filename": "GLOBAL",
    "class_name": "ALL",
    "status": "summary",
    "Ground Truth Pixels": overall["Ground Truth Pixels"],
    "Prediction Pixels": overall["Prediction Pixels"],
    "Pred/Ground Ratio": overall["Prediction Pixels"] / overall["Ground Truth Pixels"] if overall["Ground Truth Pixels"] > 0 else 0,
    "Intersection": overall["Intersection"],
    "IoU": df["IoU"].mean(),
    "Dice": df["Dice"].mean(),
    "GT_objects_org": overall["GT_objects_org"],
    "GT_objects_filt": overall["GT_objects_filt"],
    "Pred_objects_org": overall["Pred_objects_org"],
    "Pred_objects_filt": overall["Pred_objects_filt"],
    "TP": overall_TP,
    "FP": overall_FP,
    "FN": overall_FN,
    "Precision": overall_precision,
    "Recall": overall_recall,
    "F1": overall_f1,
    "Accuracy": overall_acc
}

# ================= FINAL SAVE =================
df = pd.concat([df, pd.DataFrame(classwise_results + [summary])], ignore_index=True)
df.to_csv(SAVE_CSV, index=False)
print(f"✅ Results saved to {SAVE_CSV}")
