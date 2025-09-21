import os
import cv2
import numpy as np

# --- CONFIG ---
image_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\IMAGES_4030"
mask_dir = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\MASKS_4030"
predicted_dir = None #r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\OG_DATASET_CONCRETE\AnnotationMasks"  # or "None"
output_video = r"D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\4030_4040\output_videoq.mp4"
frame_width = 419
frame_height = 1024
fps = 4

# Check if predicted folder exists and has images
use_pred = predicted_dir is not None and os.path.exists(predicted_dir) \
           and any(f.lower().endswith(('.png','.jpg','.jpeg')) for f in os.listdir(predicted_dir))

# Get image files
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

# Determine video width
video_width = frame_width*4 if use_pred else frame_width*3
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_width, frame_height))

for fname in image_files:
    img_path = os.path.join(image_dir, fname)

    # --- Original mask ---
    mask_path = None
    for ext in ['.png','.jpg','.jpeg']:
        candidate = os.path.join(mask_dir, os.path.splitext(fname)[0] + ext)
        if os.path.exists(candidate):
            mask_path = candidate
            break

    # --- Predicted mask ---
    pred_path = None
    if use_pred:
        for ext in ['.png','.jpg','.jpeg']:
            candidate = os.path.join(predicted_dir, os.path.splitext(fname)[0] + ext)
            if os.path.exists(candidate):
                pred_path = candidate
                break

    # --- Read original image ---
    img = cv2.imread(img_path)
    img = cv2.resize(img, (frame_width, frame_height))

    # --- Read original mask ---
    if mask_path:
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (frame_width, frame_height))
        overlay_mask = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        display_mask = mask
    else:
        display_mask = np.zeros((frame_height, frame_width, 3), np.uint8)
        overlay_mask = display_mask.copy()
        overlay_mask = img.copy()
        cv2.putText(display_mask, "Mask Not Found", (50, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # --- Predicted overlay ---
    if use_pred:
        if pred_path:
            pred = cv2.imread(pred_path)
            pred = cv2.resize(pred, (frame_width, frame_height))
            overlay_pred = cv2.addWeighted(img, 0.7, pred, 0.3, 0)
            display_pred = pred
        else:
            display_pred = np.zeros((frame_height, frame_width, 3), np.uint8)
            overlay_pred = display_pred.copy()
            cv2.putText(display_pred, "Predicted Not Found", (50, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # --- Stack images ---
    if use_pred:
        stacked = np.hstack([img, display_mask, display_pred, overlay_pred])
    else:
        stacked = np.hstack([img, display_mask, overlay_mask])

    # --- Add filename ---
    cv2.putText(stacked, fname, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1)

    # --- Display ---
    cv2.imshow("Comparison", stacked)
    key = cv2.waitKey(50) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("Exited by user")
        break

    # --- Write video ---
    out.write(stacked)

out.release()
cv2.destroyAllWindows()
print("Video saved to:", output_video)
