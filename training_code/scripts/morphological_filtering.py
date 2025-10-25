import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Load image
img_path = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0005791.png"
img_path = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0005605.png"
img_path = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\ACCEPTED_MASKS\AMRAVTI-TALEGAON_2025-06-14_06-38-51_SECTION-1_IMG_0004067.png"
img = cv2.imread(img_path)

# Color map
COLOR_MAP = {
    (255, 0, 0): "Alligator",
    (0, 0, 255): "Transverse Crack",
    (0, 255, 0): "Longitudinal Crack",
}

def find_endpoints(contour):
    """Find the two farthest points in a contour"""
    pts = contour.reshape(-1, 2)
    max_dist = 0
    endpoints = (pts[0], pts[0])
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist = d
                endpoints = (pts[i], pts[j])
    return endpoints

def join_segments(mask, radius=250, line_width=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    endpoints = []
    for cnt in contours:
        p1, p2 = find_endpoints(cnt)
        endpoints.append((p1, p2))

    joined = mask.copy()
    for (a1, a2), (b1, b2) in combinations(endpoints, 2):
        for p, q in [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]:
            if np.linalg.norm(p - q) < radius:
                cv2.line(joined, tuple(p), tuple(q), 255, line_width)
    return joined

# Create joined masks
new_img = np.zeros_like(img)
for color, name in COLOR_MAP.items():
    mask = cv2.inRange(img, np.array(color), np.array(color))
    joined_mask = join_segments(mask, radius=35, line_width=2)
    new_img[joined_mask > 0] = color

# Show comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 12))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Mask")
ax[0].axis("off")

ax[1].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
ax[1].set_title("Joined (Geometric) Mask")
ax[1].axis("off")

plt.tight_layout()
plt.show()
