import os
import torch
from ultralytics import YOLO
from segmentation_architecture.UnetPP import UNetPP  # Adjust import


class ONNXExporter:
    def __init__(self, model_type: str, model_path: str, onnx_path: str,
                 input_size=(1024, 419), class_names=None, device=None):
        """
        :param model_type: 'yolo' or 'unet'
        :param model_path: path to .pt/.pth file
        :param onnx_path: path to save .onnx file
        :param input_size: input resolution (H, W)
        :param class_names: list of class names (for UNet++)
        :param device: torch.device
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.height, self.width = input_size
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def export(self):
        if self.model_type == "yolo":
            self._export_yolo()
        elif self.model_type == "unet":
            self._export_unet()
        else:
            raise ValueError("Invalid model_type. Use 'yolo' or 'unet'.")

    def _export_yolo(self):
        print(f"Loading YOLO model from {self.model_path}...")
        model = YOLO(self.model_path)

        print(f"Exporting YOLO to {self.onnx_path}...")
        model.export(format='onnx', opset=12, simplify=False)
        print("✅ YOLO ONNX export complete.")

        # ---- Save classes.txt ----
        if hasattr(model, "names") and isinstance(model.names, dict):
            class_names = [name for _, name in sorted(model.names.items())]
            self._save_class_names(class_names)

    def _detect_unet_num_classes(self, state_dict):
        """Try to infer num_classes from checkpoint."""
        if isinstance(state_dict, dict) and "num_classes" in state_dict:
            return state_dict["num_classes"]

        if "model" in state_dict and isinstance(state_dict["model"], dict) and "num_classes" in state_dict["model"]:
            return state_dict["model"]["num_classes"]

        for k, v in state_dict.items():
            if any(x in k for x in ["final", "classifier", "out_conv"]):
                if len(v.shape) == 4:  # Conv2d weights
                    return v.shape[0]  # out_channels = num_classes
        return 1

    def _export_unet(self):
        print(f"Loading UNet++ model from {self.model_path}...")

        ckpt = torch.load(self.model_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        num_classes = self._detect_unet_num_classes(state_dict)
        print(f"Detected num_classes = {num_classes}")

        model_seg = UNetPP(in_channels=3, num_classes=num_classes).to(self.device)
        model_seg.load_state_dict(state_dict, strict=False)
        model_seg.eval()

        dummy_input = torch.randn(1, 3, self.height, self.width).to(self.device)

        print(f"Exporting UNet++ to {self.onnx_path}...")
        torch.onnx.export(
            model_seg,
            dummy_input,
            self.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )
        print("✅ UNet++ ONNX export complete.")

        # ---- Save classes.txt (if provided) ----
        if self.class_names:
            if len(self.class_names) != num_classes:
                print(f"⚠️ Warning: Provided {len(self.class_names)} class names, "
                      f"but model has {num_classes} outputs")
            self._save_class_names(self.class_names)

    def _save_class_names(self, class_names):
        """Save class names to classes.txt next to ONNX."""
        classes_txt_path = os.path.join(os.path.dirname(self.onnx_path), "classes.txt")
        with open(classes_txt_path, "w") as f:
            f.write("\n".join(class_names))
        print(f"✅ Saved class names to {classes_txt_path}")


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # YOLO Export
    exporter_yolo = ONNXExporter(
        model_type="yolo",
        model_path=r"C:\Users\Admin\Code\survey-analytic\ai_model\guide_rcc_metal_median.pt",
        onnx_path=r"C:\Users\Admin\Code\survey-analytic\ai_model\guide_rcc_metal_median.onnx"
    )
    exporter_yolo.export()

    # UNet++ Export with class names
    unet_classes = ["Background", "Alligator", "Longitudinal", "Transverse", "Multiple", "Joint Seal"]
    exporter_unet = ONNXExporter(
        model_type="unet",
        model_path=r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_hybrid\UNET_V2_best_epoch117_dice0.729.pth",
        onnx_path=r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_hybrid\UnetPP_14aug.onnx",
        input_size=(1024, 419),
        # class_names=unet_classes,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    exporter_unet.export()
