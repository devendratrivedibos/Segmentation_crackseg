import torch
import os
from models.segformer.segformer import SegFormer  # Adjust path if it's elsewhere

# ---- Configuration ----
pth_path = r'D:\Devendra_Files\CrackSegFormer-main\segformer\20250805-194612-best_model.pth'  # Your .pth file
onnx_path = r'D:\Devendra_Files\CrackSegFormer-main\segformer\segformer.onnx'  # Where to save .onnx
num_classes = 2  # Background + your target classes (adjust as needed)
phi = 'b0'  # Or 'b0', 'b1', etc.
img_size = 512  # Input size expected by your model (or as used in training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Model Preparation ----
model = SegFormer(num_classes=num_classes, phi=phi, pretrained=False)
model = model.to(device)
model.eval()

# If loading weights (just state_dict)
state_dict = torch.load(pth_path, map_location=device)
if 'model' in state_dict:
    state_dict = state_dict['model']  # If checkpoint was saved with 'model' key
model.load_state_dict(state_dict, strict=False)

# ---- Example Input ----
dummy_input = torch.randn(1, 3, img_size, img_size).to(device)  # Batch size 1, 3 channels

# ---- Export to ONNX ----
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,   # 11 or newer is widely supported
    do_constant_folding=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
)

print(f"ONNX export complete: {onnx_path}")


import onnx

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")


import onnxruntime as ort
import numpy as np
import cv2

# 1. Load the ONNX model
session = ort.InferenceSession(onnx_path)

# 2. Prepare an input image (match your training resolution!)
img_path = "D:\cracks\Semantic-Segmentation of pavement distress dataset\Combined\Devendra\A_T_1_rangeDataFiltered-0000272-_crack.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))  # Adjust size as needed

# 3. Normalize as during training (use your mean/std)
img = img.astype(np.float32) / 255.0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mean = np.array(mean)
std = np.array(std)
img = (img - mean) / std

# 4. HWC to CHW and add batch dimension
input_tensor = np.transpose(img, (2, 0, 1))[None, :, :, :].astype(np.float32)  # Shape: [1, 3, H, W]

# 5. Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 6. Run inference
output = session.run([output_name], {input_name: input_tensor})[0]

# 7. Process output (get mask via argmax)
# output shape: [1, num_classes, H, W]
mask = np.argmax(output[0], axis=0).astype(np.uint8)  # Shape: [H, W]

# 8. (Optional) Map mask to colors for visualization
# Example: for 2 classes, background=0 (black), crack=1 (white)
colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
output_img = colors[mask]

cv2.imwrite("D:/segformer_onnx_prediction.png", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

print("Inference completed. Output mask saved as segformer_onnx_prediction.png")


