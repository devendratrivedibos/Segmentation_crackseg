import os
import cv2
import pandas as pd

# ================== PATHS ==================
folder_path = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5\process_distress_results_4oct_latest"
output_folder = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5\process_distress_results_15oct_latest"
os.makedirs(output_folder, exist_ok=True)

pothole_csv_path = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5\reportNew\csv_reports\potholes.csv"
patches_csv_path = r"Y:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-5\reportNew\csv_reports\patches.csv"

pothole_df = pd.read_csv(pothole_csv_path)
patches_df = pd.read_csv(patches_csv_path)

# ================== COLORS ==================
pothole_color = (139, 69, 19)   # Brown
patches_color = (255, 165, 0)   # Orange

# Convert RGB → BGR for OpenCV
pothole_color = tuple(reversed(pothole_color))
patches_color = tuple(reversed(patches_color))
# ================== ORIGINAL IMAGE SIZE ==================
orig_pothole_width = 5120
orig_pothole_height = 2000

orig_patches_width = 1280
orig_patches_height = 720
# ================== FUNCTIONS ==================
def parse_bbox(row):
    """Safely parse bbox columns into list of (x, y, w, h) tuples"""
    try:
        if row['X Coordinates'] in ('', '[]', None):
            return []
        xs = row['X Coordinates'].replace('[', '').replace(']', '').split(',')
        ys = row['Y Coordinates'].replace('[', '').replace(']', '').split(',')
        ws = row['Widths'].replace('[', '').replace(']', '').split(',')
        hs = row['Heights'].replace('[', '').replace(']', '').split(',')

        bboxes = []
        for x, y, w, h in zip(xs, ys, ws, hs):
            if x.strip() and y.strip() and w.strip() and h.strip():
                bboxes.append((float(x), float(y), float(w), float(h)))
        return bboxes
    except Exception as e:
        print(f"[WARN] Failed parsing bbox: {e}")
        return []

# Create bbox column for both CSVs
pothole_df['bbox'] = pothole_df.apply(parse_bbox, axis=1)
patches_df['bbox'] = patches_df.apply(parse_bbox, axis=1)

# Drop rows where bbox is empty
pothole_df = pothole_df[pothole_df['bbox'].apply(lambda x: len(x) > 0)]
patches_df = patches_df[patches_df['bbox'].apply(lambda x: len(x) > 0)]



# ================== PROCESS EACH REGION ==================
unique_regions = sorted(set(pothole_df['Region_index'].unique()) |
                        set(patches_df['Region_index'].unique()))

for region_index in unique_regions:
    img_files = [f for f in os.listdir(folder_path)
                 if str(region_index) in f and f.lower().endswith(('.jpg', '.png'))]

    if not img_files:
        print(f"[SKIP] No image found for Region_index {region_index}")
        continue

    img_name = img_files[0]
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.flip(img, 0)
    if img is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        continue

    new_h, new_w = img.shape[:2]
    pothole_scale_x = new_w / orig_pothole_width
    pothole_scale_y = new_h / orig_pothole_height

    patches_scale_x = new_w / orig_patches_width
    patches_scale_y = new_h / orig_patches_height

    # ---- Potholes ----
    df_poth = pothole_df[pothole_df['Region_index'] == region_index]
    for bbox_list in df_poth['bbox']:
        for (x, y, w, h) in bbox_list:
            x, y, w, h = int(x * pothole_scale_x), int(y * pothole_scale_y), int(w * pothole_scale_x), int(h * pothole_scale_y)
            cv2.rectangle(img, (x, y), (w, h), pothole_color, 2)

    # ---- Patches ----
    df_patch = patches_df[patches_df['Region_index'] == region_index]
    for bbox_list in df_patch['bbox']:
        for (x, y, w, h) in bbox_list:
            x, y, w, h = int(x * patches_scale_x), int(y * patches_scale_y)-50, int(w * patches_scale_x), int(h * patches_scale_y-50)
            cv2.rectangle(img, (x, y), (w, h), patches_color, -1)  # filled rectangle

    # Save annotated image
    output_path = os.path.join(output_folder, img_name)

    cv2.imwrite(output_path, img)
    print(f"[OK] Region {region_index}: saved {output_path}")

print("✅ Annotation completed for potholes (brown) and patches (orange, filled).")
