import os
import cv2
import numpy as np

class ImageMaskOverlay:
    def __init__(self, image_dir, mask_dir, predicted_dir=None, output_video="output.avi", size=(1024, 419)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.predicted_dir = predicted_dir
        self.output_video = output_video
        self.size = size

        # Check if prediction is used
        self.use_pred = predicted_dir is not None

        # Collect all filenames (union of images, masks, predictions)
        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

        if self.use_pred:
            pred_files = {os.path.splitext(f)[0]: f for f in os.listdir(predicted_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        else:
            pred_files = {}

        self.all_keys = sorted(set(img_files.keys()) | set(mask_files.keys()) | set(pred_files.keys()))
        self.img_files, self.mask_files, self.pred_files = img_files, mask_files, pred_files

        # Video writer setup
        if self.use_pred:
            frame_width = size[1] * 4  # Image + Mask + Pred + Overlay
        else:
            frame_width = size[1] * 3  # Image + Mask + Overlay
        frame_height = size[0]

        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_video, self.fourcc, 1, (frame_width, frame_height))

    def _load_image(self, path, text="Not Found", color=(0, 0, 255)):
        if path and os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.resize(img, self.size[::-1])
        else:
            img = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            cv2.putText(img, text, (30, self.size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return img

    def _overlay(self, base, mask, color=(0, 0, 255)):
        """Overlay mask on base image with given color."""
        return cv2.addWeighted(base, 0.7, mask, 0.3, 0)

    def process(self):
        for key in self.all_keys:
            img_path = os.path.join(self.image_dir, self.img_files.get(key, "")) if key in self.img_files else None
            mask_path = os.path.join(self.mask_dir, self.mask_files.get(key, "")) if key in self.mask_files else None
            pred_path = os.path.join(self.predicted_dir,
                                     self.pred_files.get(key, "")) if self.use_pred and key in self.pred_files else None

            # --- Load Image ---
            img = self._load_image(img_path, "Image Not Found", (255, 255, 255))

            # --- Load Mask ---
            if mask_path:
                mask = self._load_image(mask_path)
                overlay_mask = self._overlay(img, mask, color=(0, 0, 255))
            else:
                mask = self._load_image(None, "Mask Not Found", (0, 0, 255))
                overlay_mask = img.copy()  # show actual if mask missing

            # --- Predicted Mask ---
            if self.use_pred:
                if pred_path:
                    pred = self._load_image(pred_path)
                    overlay_pred = self._overlay(img, pred, color=(0, 255, 0))
                else:
                    pred = self._load_image(None, "Predicted Not Found", (0, 255, 0))
                    overlay_pred = img.copy()  # show actual if pred missing

                stacked = np.hstack([img, mask, pred, overlay_pred])
            else:
                stacked = np.hstack([img, mask, overlay_mask])

            # --- Put filename on top ---
            cv2.putText(stacked, key, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            # --- Write to video ---
            self.writer.write(stacked)

            # --- Show live ---
            cv2.imshow("Preview", stacked)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        self.writer.release()
        cv2.destroyAllWindows()


# ------------------ USAGE ------------------
if __name__ == "__main__":

    # --- CONFIG ---
    predicted_dir = None

    image_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1/process_distress_og"
    mask_dir = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1/process_distress_results_skeleton"
    output_video = r"Z:\NHAI_Amaravati_Data\AMRAVTI-TALEGAON_2025-06-14_06-38-51\SECTION-1\Amrawati-Talegaon_segmentation_s1.mp4"
    processor = ImageMaskOverlay(image_dir, mask_dir, predicted_dir, output_video)
    processor.process()
