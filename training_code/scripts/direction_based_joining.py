import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# =====================================
# CONFIG
# =====================================
img_path = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\ACCEPTED_IMAGES\mask_color_bgr.png"
img = cv2.imread(img_path)

COLOR_MAP = {
    (0, 0, 0): (0, "Background"),
    (255, 0, 0): (1, "Alligator"),
    (0, 0, 255): (2, "Transverse Crack"),
    (0, 255, 0): (3, "Longitudinal Crack"),
    (139, 69, 19): (4, "Pothole"),
    (255, 165, 0): (5, "Patches"),
}

# =====================================
# UTILITIES
# =====================================
def find_endpoints(contour):
    """Return two farthest points in contour (endpoints)."""
    pts = contour.reshape(-1, 2)
    max_dist = 0
    endpoints = (pts[0], pts[0])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist = d
                endpoints = (pts[i], pts[j])
    return endpoints


def join_directional(mask, crack_type, radius=5, line_width=2, proximity_thresh=20):
    """Join cracks geometrically in direction-aware + proximity-aware fashion."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    endpoints = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        p1, p2 = find_endpoints(cnt)
        endpoints.append((p1, p2))

    joined = mask.copy()

    for (a1, a2), (b1, b2) in combinations(endpoints, 2):
        for p, q in [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]:
            dx, dy = abs(int(p[0]) - int(q[0])), abs(int(p[1]) - int(q[1]))
            dist = np.hypot(dx, dy)

            connected = False
            if crack_type == "Longitudinal Crack":
                if dx <= radius and dy <= 25:  # vertical direction
                    connected = True
            elif crack_type == "Transverse Crack":
                if dy <= radius and dx <= 25:  # horizontal direction
                    connected = True
            elif crack_type == "Alligator":
                if dist <= radius:  # general small gaps
                    connected = True

            # NEW: join if endpoints are close enough, regardless of direction
            if not connected and dist <= proximity_thresh:
                connected = True

            if connected:
                cv2.line(joined, tuple(p), tuple(q), 255, line_width)

    return joined


def colorize_prediction(idx_map, original_img):
    """Convert class index map → RGB image, preserve unknown classes."""
    output = original_img.copy()
    for rgb, (idx, _) in COLOR_MAP.items():
        output[idx_map == idx] = rgb
    return output


# =====================================
# STEP 1: CONVERT RGB IMAGE TO CLASS-INDEX MAP
# =====================================
# Initialize with 255 = unknown (preserve)
idx_map = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)

for rgb, (idx, _) in COLOR_MAP.items():
    mask = np.all(img == np.array(rgb, dtype=np.uint8), axis=-1)
    idx_map[mask] = idx

# =====================================
# STEP 2: PROCESS CRACK CLASSES
# =====================================
joined_idx_map = idx_map.copy()

for rgb, (idx, name) in COLOR_MAP.items():
    if name in ["Longitudinal Crack", "Transverse Crack", "Alligator"]:
        binary_mask = (idx_map == idx).astype(np.uint8) * 255
        joined_mask = join_directional(binary_mask, name, radius=25, line_width=2)
        joined_idx_map[joined_mask > 0] = idx  # update class index map

# =====================================
# STEP 3: RECOLOR FINAL OUTPUT
# =====================================
joined_img = colorize_prediction(joined_idx_map, img)

# =====================================
# DISPLAY
# =====================================
fig, ax = plt.subplots(1, 2, figsize=(12, 12))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Mask")
ax[0].axis("off")

ax[1].imshow(cv2.cvtColor(joined_img, cv2.COLOR_BGR2RGB))
ax[1].set_title("Direction-Aware Joined Mask")
ax[1].axis("off")

plt.tight_layout()
plt.show()
