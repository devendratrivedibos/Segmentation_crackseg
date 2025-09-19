import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================== CONFIG ==================
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
    (255, 100, 203): (9, "Sealed Joint - T"),
    (199, 21, 133): (10, "Sealed Joint - L"),
    (128, 0, 128): (11, "Punchout"),
    (112, 102, 255): (12, "Popout"),
    (255, 255, 255): (13, "Unclassified"),
    (255, 215, 0): (14, "Cracking"),
}

GT_DIR = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\SPLITTED\TEST\MASKS"   # ground truth folder
PRED_DIR = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\SPLITTED\TEST\RESULTS" # predicted masks
SAVE_CSV = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\ASPHALT_ACCEPTED\SPLITTED\TEST\mask_metrics.csv"


# ================== HELPERS ==================
def color_to_index(mask, color_map):
    """Convert RGB mask to index mask"""
    h, w, _ = mask.shape
    index_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, (idx, _) in color_map.items():
        match = np.all(mask == rgb, axis=-1)
        index_mask[match] = idx
    return index_mask


# ---------- Segmentation Metrics (pixel-level) ----------
def compute_classwise_overlap(gt, pred, color_map, ignore_background=True):
    """Return per-class segmentation IoU/Dice (only for classes in GT or Pred)"""
    stats = []
    classes_in_use = np.unique(np.concatenate([np.unique(gt), np.unique(pred)]))

    for c in classes_in_use:
        if ignore_background and c == 0:
            continue

        gt_mask   = (gt == c)
        pred_mask = (pred == c)

        gt_count   = gt_mask.sum()
        pred_count = pred_mask.sum()
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union        = np.logical_or(gt_mask, pred_mask).sum()

        iou  = intersection / (union + 1e-7) if union > 0 else 0.0
        dice = (2 * intersection) / (gt_count + pred_count + 1e-7) if (gt_count+pred_count)>0 else 0.0

        class_name = [name for _, (idx, name) in color_map.items() if idx == c][0]

        stats.append({
            "class_name": class_name,
            "GT_pixels": int(gt_count),
            "Pred_pixels": int(pred_count),
            "Intersection": int(intersection),
            "IoU": iou,
            "Dice": dice
        })
    return stats


# ---------- Detection Metrics (object-level) ----------
def get_objects(mask, class_id):
    """Extract connected components for a given class_id mask"""
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    objects = []
    for i in range(1, num_labels):  # skip background
        obj_mask = (labels == i).astype(np.uint8)
        objects.append(obj_mask)
    return objects

def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0

def compute_detection_metrics(gt_mask, pred_mask, color_map, iou_thresh=0.5, ignore_background=True):
    """Return detection stats per class (TP/FP/FN, Precision, Recall, F1)"""
    stats = []
    classes_in_use = np.unique(np.concatenate([np.unique(gt_mask), np.unique(pred_mask)]))

    for c in classes_in_use:
        if ignore_background and c == 0:
            continue

        gt_objects = get_objects(gt_mask, c)
        pred_objects = get_objects(pred_mask, c)

        matched_gt = set()
        matched_pred = set()

        for gi, g in enumerate(gt_objects):
            for pi, p in enumerate(pred_objects):
                iou = compute_iou(g, p)
                if iou >= iou_thresh:
                    matched_gt.add(gi)
                    matched_pred.add(pi)

        TP = len(matched_gt)
        FN = len(gt_objects) - TP
        FP = len(pred_objects) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_name = [name for _, (idx, name) in color_map.items() if idx == c][0]

        stats.append({
            "class_name": class_name,
            "GT_objects": len(gt_objects),
            "Pred_objects": len(pred_objects),
            "TP": TP, "FP": FP, "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })
    return stats


# ================== MAIN LOOP ==================
all_results = []

for fname in tqdm(os.listdir(GT_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    gt_path = os.path.join(GT_DIR, fname)
    pred_path = os.path.join(PRED_DIR, fname)

    if not os.path.exists(pred_path):
        print(f"⚠️ Missing prediction for {fname}")
        continue

    # Read images
    gt_img   = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
    pred_img = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)

    # Convert to index masks
    gt_mask   = color_to_index(gt_img, COLOR_MAP)
    pred_mask = color_to_index(pred_img, COLOR_MAP)

    # --- Pixel-level metrics ---
    seg_stats = compute_classwise_overlap(gt_mask, pred_mask, COLOR_MAP, ignore_background=True)

    # --- Object-level metrics ---
    det_stats = compute_detection_metrics(gt_mask, pred_mask, COLOR_MAP, iou_thresh=0.5, ignore_background=True)

    # Merge results per class
    class_names = {s["class_name"] for s in seg_stats} | {d["class_name"] for d in det_stats}

    for cname in class_names:
        seg = next((s for s in seg_stats if s["class_name"] == cname), {})
        det = next((d for d in det_stats if d["class_name"] == cname), {})

        all_results.append({
            "filename": fname,
            "class_name": cname,

            # Segmentation
            "GT_pixels": seg.get("GT_pixels", 0),
            "Pred_pixels": seg.get("Pred_pixels", 0),
            "Intersection": seg.get("Intersection", 0),
            "IoU": seg.get("IoU", 0.0),
            "Dice": seg.get("Dice", 0.0),

            # Detection
            "GT_objects": det.get("GT_objects", 0),
            "Pred_objects": det.get("Pred_objects", 0),
            "TP": det.get("TP", 0),
            "FP": det.get("FP", 0),
            "FN": det.get("FN", 0),
            "Precision": det.get("Precision", 0.0),
            "Recall": det.get("Recall", 0.0),
            "F1": det.get("F1", 0.0),
        })

# Save results
df = pd.DataFrame(all_results)
df.to_csv(SAVE_CSV, index=False)
print(f"✅ Results saved to {SAVE_CSV}")
