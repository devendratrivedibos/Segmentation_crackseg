import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.morphology import skeletonize

# ================== CONFIG ==================
root_dir = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51"

GT_DIRS = [
    os.path.join(root_dir, "SECTION-1", "ACCEPTED_MASKS"),
    # os.path.join(root_dir, "SECTION-2", "ACCEPTED_MASKS"),
    # os.path.join(root_dir, "SECTION-3", "ACCEPTED_MASKS"),
    # os.path.join(root_dir, "SECTION-4", "ACCEPTED_MASKS"),
    # os.path.join(root_dir, "SECTION-5", "ACCEPTED_MASKS"),
]

PRED_DIRS = [
    # os.path.join(root_dir, "SECTION-1", "process_distress_results_4oct_latest"),
    os.path.join(root_dir, "SECTION-1", "process_distress_results"),
    # os.path.join(root_dir, "SECTION-2", "process_distress_results"),
    # os.path.join(root_dir, "SECTION-3", "process_distress_results"),
    # os.path.join(root_dir, "SECTION-4", "process_distress_results"),
    # os.path.join(root_dir, "SECTION-5", "process_distress_results"),
]

SAVE_CSV = os.path.join(root_dir,"SECTION-1", "AMRAWATI-TALEGAON_mask_metrics.csv")

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

CRACK_CLASSES = {"Alligator", "Transverse Crack", "Longitudinal Crack"}

FILTER_PIXELS = {
    "Pothole": [0, 500, 1000, 1500],
    # "Alligator": [0, 50, 100],  # example thresholds for crack lengths
    # "Transverse Crack": [0, 20, 50],
    # "Longitudinal Crack": [0, 20, 50],
    # "Multiple Crack": [0, 20, 50],
    # "Cracking": [0, 50, 100],
}

# ================== LOAD FILE PATHS ==================
gt_files = {}
for gt_dir in GT_DIRS:
    for f in os.listdir(gt_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            gt_files[f] = os.path.join(gt_dir, f)

pred_files = {}
for pred_dir in PRED_DIRS:
    for f in os.listdir(pred_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and f not in pred_files:
            pred_files[f] = os.path.join(pred_dir, f)

all_files = sorted(set(gt_files) | set(pred_files))
all_results = []

# ================== HELPERS ==================
lut = np.zeros((256, 256, 256), dtype=np.uint8)
for rgb, (idx, _) in COLOR_MAP.items():
    lut[rgb] = idx

def color_to_index(mask):
    return lut[mask[..., 0], mask[..., 1], mask[..., 2]]

def morphology_operations(mask, width_kernel_size=15, height_kernel_size=30):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_kernel_size, height_kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed

def get_objects(mask, class_id, min_pixels=50):
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    objects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_pixels:
            obj_mask = (labels == i).astype(np.uint8)
            objects.append(obj_mask)
    return objects

def measure_length(binary_mask, class_name):
    """Skeletonize cracks and return length; otherwise count nonzero pixels"""
    if binary_mask.sum() == 0:
        return 0
    if class_name in CRACK_CLASSES:
        skel = skeletonize(binary_mask > 0)
        return int(np.count_nonzero(skel))
    else:
        return int(np.count_nonzero(binary_mask))


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + 1e-7) if union > 0 else 0.0


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

        # Length metrics
        gt_len = measure_length(gt_mask, class_name)
        pred_len = measure_length(pred_mask, class_name)
        length_ratio = pred_len / gt_len if gt_len > 0 else 0

        stats.append({
            "class_name": class_name,
            "Ground Truth Pixels": int(gt_count),
            "Prediction Pixels": int(pred_count),
            "Pred/Ground Ratio": pred_count / gt_count if gt_count > 0 else 0,
            "Intersection": int(inter),
            "IoU": inter / (union + 1e-7) if union > 0 else 0.0,
            "Dice": (2 * inter) / (gt_count + pred_count + 1e-7) if (gt_count + pred_count) > 0 else 0.0,
            "GT_length": gt_len,
            "Pred_length": pred_len,
            "Length_ratio": length_ratio
        })
    return stats

def compute_detection_metrics_multi_threshold(gt_mask, pred_mask, color_map, iou_thresh=0.1):
    all_stats = []
    classes_in_use = np.unique(np.concatenate([np.unique(gt_mask), np.unique(pred_mask)]))

    for c in classes_in_use:
        if c == 0:
            continue
        class_name = [name for _, (idx, name) in color_map.items() if idx == c][0]
        if class_name in IGNORE_CLASSES:
            continue

        gt_class_mask = morphology_operations((gt_mask == c).astype(np.uint8))
        pred_class_mask = morphology_operations((pred_mask == c).astype(np.uint8))

        stats = {"class_name": class_name}

        thresholds = FILTER_PIXELS.get(class_name, [0])
        for th in thresholds:
            gt_objects = get_objects(gt_class_mask, 1, min_pixels=th)
            pred_objects = get_objects(pred_class_mask, 1, min_pixels=th)
            matched_gt = set()
            matched_pred = set()
            for gi, g in enumerate(gt_objects):
                for pi, p in enumerate(pred_objects):
                    if compute_iou(g, p) >= iou_thresh:
                        matched_gt.add(gi)
                        matched_pred.add(pi)

            TP = len(matched_gt)
            FN = len(gt_objects) - TP
            FP = len(pred_objects) - TP
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Compute lengths per threshold
            gt_len = sum(measure_length(obj, class_name) for obj in gt_objects)
            pred_len = sum(measure_length(obj, class_name) for obj in pred_objects)
            length_ratio = pred_len / gt_len if gt_len > 0 else 0

            stats.update({
                f"GT_objects_filt_{th}": len(gt_objects),
                f"Pred_objects_filt_{th}": len(pred_objects),
                f"TP_{th}": TP,
                f"FP_{th}": FP,
                f"FN_{th}": FN,
                f"Precision_{th}": precision,
                f"Recall_{th}": recall,
                f"F1_{th}": f1,
                f"GT_length_{th}": gt_len,
                f"Pred_length_{th}": pred_len,
                f"Length_ratio_{th}": length_ratio
            })
        all_stats.append(stats)
    return all_stats

# ================== MAIN LOOP ==================
for fname in tqdm(all_files):
    gt_path, pred_path = gt_files.get(fname), pred_files.get(fname)
    gt_mask = color_to_index(cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)) if gt_path else np.zeros((1024, 419), np.uint8)
    pred_mask = color_to_index(cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)) if pred_path else np.zeros_like(gt_mask)

    gt_mask = cv2.resize(gt_mask, (419, 1024), interpolation=cv2.INTER_NEAREST)
    pred_mask = cv2.resize(pred_mask, (419, 1024), interpolation=cv2.INTER_NEAREST)

    seg_stats = compute_classwise_overlap(gt_mask, pred_mask)
    det_stats = compute_detection_metrics_multi_threshold(gt_mask, pred_mask, COLOR_MAP)

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
            "GT_length": seg.get("GT_length", 0),
            "Pred_length": seg.get("Pred_length", 0),
            "Length_ratio": seg.get("Length_ratio", 0),
            **det
        })

# ================= CLASS-WISE SUMMARY =================
df = pd.DataFrame(all_results)
classwise_results = []

for cname, g in df.groupby("class_name"):
    class_dict = {
        "filename": "GLOBAL",
        "class_name": cname,
        "status": "class_summary",
        "Ground Truth Pixels": g["Ground Truth Pixels"].sum(),
        "Prediction Pixels": g["Prediction Pixels"].sum(),
        "Intersection": g["Intersection"].sum(),
        "IoU": g["IoU"].mean(),
        "Dice": g["Dice"].mean(),
        "GT_length": g["GT_length"].sum(),
        "Pred_length": g["Pred_length"].sum(),
        "Length_ratio": g["Pred_length"].sum() / g["GT_length"].sum() if g["GT_length"].sum() > 0 else 0
    }

    thresholds = FILTER_PIXELS.get(cname, [0])
    for th in thresholds:
        class_dict[f"GT_objects_filt_{th}"] = g.get(f"GT_objects_filt_{th}", pd.Series(dtype=float)).sum()
        class_dict[f"Pred_objects_filt_{th}"] = g.get(f"Pred_objects_filt_{th}", pd.Series(dtype=float)).sum()
        class_dict[f"GT_length_{th}"] = g.get(f"GT_length_{th}", pd.Series(dtype=float)).sum()
        class_dict[f"Pred_length_{th}"] = g.get(f"Pred_length_{th}", pd.Series(dtype=float)).sum()
        class_dict[f"Length_ratio_{th}"] = (class_dict[f"Pred_length_{th}"] / class_dict[f"GT_length_{th}"]
                                            if class_dict[f"GT_length_{th}"] > 0 else 0)
        TP = g.get(f"TP_{th}", pd.Series(dtype=float)).sum()
        FP = g.get(f"FP_{th}", pd.Series(dtype=float)).sum()
        FN = g.get(f"FN_{th}", pd.Series(dtype=float)).sum()
        class_dict.update({
            f"TP_{th}": TP,
            f"FP_{th}": FP,
            f"FN_{th}": FN,
            f"Precision_{th}": TP / (TP + FP) if (TP + FP) > 0 else 0,
            f"Recall_{th}": TP / (TP + FN) if (TP + FN) > 0 else 0,
            f"F1_{th}": 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0
        })
    classwise_results.append(class_dict)

# ================= OVERALL SUMMARY =================
overall_dict = {"filename": "GLOBAL", "class_name": "ALL", "status": "summary"}
numeric_cols = [c for c in df.columns if df[c].dtype in [np.int64, np.float64]]
overall = df[numeric_cols].sum()
overall_dict.update(overall.to_dict())

df_final = pd.concat([df, pd.DataFrame(classwise_results + [overall_dict])], ignore_index=True)
df_final.to_csv(SAVE_CSV, index=False)
print(f"✅ Results saved to {SAVE_CSV}")
