import os
import cv2
import numpy as np
import psycopg2
from shapely.geometry import Polygon
from shapely import wkt

# -------------------- CONFIG --------------------
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

DB_CONFIG = {
    "dbname": "Segmentation",
    "user": "bosadmin",
    "password": "admin@321",
    "host": "192.168.1.14",
    "port": 5432
}

# -------------------- DB CONNECTION --------------------
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# -------------------- FUNCTIONS --------------------

def safe_load_polygon(polygon_wkt):
    """Load WKT polygon, fixing if not closed"""
    try:
        return wkt.loads(polygon_wkt)
    except Exception:
        coords = polygon_wkt.strip().replace("POLYGON((", "").replace("))", "")
        pts = [tuple(map(float, xy.split())) for xy in coords.split(",")]
        if pts[0] != pts[-1]:
            pts.append(pts[0])  # close polygon
        fixed_wkt = "POLYGON((" + ", ".join([f"{x} {y}" for x, y in pts]) + "))"
        return wkt.loads(fixed_wkt)

def save_masks_to_db(image_dir):
    """Extract polygons from masks and save into DB"""
    all_records = []

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        mask_path = os.path.join(image_dir, fname)
        mask = cv2.imread(mask_path)
        image_name = os.path.splitext(fname)[0]  # save basename only

        for color, (class_id, class_name) in COLOR_MAP.items():
            if class_id == 0:  # skip background
                continue

            # Create binary mask for this color
            binary = cv2.inRange(mask, np.array(color), np.array(color))
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 5:  # skip tiny noise
                    continue
                pts = cnt.reshape(-1, 2)
                pts_str = ", ".join([f"{x} {y}" for x, y in pts])
                polygon_wkt = f"POLYGON(({pts_str}))"
                all_records.append((image_name, class_id, class_name, polygon_wkt))

    # Insert into DB
    from psycopg2.extras import execute_batch
    execute_batch(cur,
                  "INSERT INTO image_annotations (image_name, class_id, class_name, polygon_wkt) VALUES (%s, %s, %s, %s)",
                  all_records)
    conn.commit()
    print(f"Saved {len(all_records)} polygons into DB.")


def reconstruct_images_from_db(output_dir):
    """Reconstruct mask images from DB polygons"""
    os.makedirs(output_dir, exist_ok=True)
    cur.execute("SELECT image_name, class_id, polygon_wkt FROM image_annotations ORDER BY image_name;")
    rows = cur.fetchall()

    images = {}
    for image_name, class_id, polygon_wkt in rows:
        images.setdefault(image_name, []).append((class_id, polygon_wkt))

    for image_name, items in images.items():
        height, width = 1024, 419

        mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, polygon_wkt in items:
            polygon = safe_load_polygon(polygon_wkt)
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            color = [k for k, v in COLOR_MAP.items() if v[0] == class_id]
            color = color[0] if color else (255, 255, 255)
            cv2.fillPoly(mask, [coords], color)

        cv2.imwrite(os.path.join(output_dir, f"{image_name}_reconstructed.png"), mask)

    print(f"Reconstructed {len(images)} images into {output_dir}")


# -------------------- EXAMPLE USAGE --------------------
# Step 1: Save all mask images into DB
# save_masks_to_db(r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5\ACCEPTED_MASKS")

# Step 2: Reconstruct all images from DB
reconstruct_images_from_db(r"reconstructed_masks")
# Close DB
cur.close()
conn.close()
