import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 1. Load DINOv2 backbone
# ------------------------------
def get_dinov2_backbone(name="dinov2_vitl14"):
    backbone = torch.hub.load("facebookresearch/dinov2", name)
    embed_dim = getattr(backbone, "embed_dim", 384)
    return backbone, embed_dim

# ------------------------------
# 2. Load DINOv3 backbone
# ------------------------------
def get_dinov3_backbone(name="dinov3_vitb16_lvd142m"):
    backbone = torch.hub.load("facebookresearch/dinov3", name)
    embed_dim = getattr(backbone, "embed_dim", 384)
    return backbone, embed_dim

# ------------------------------
# 3. ASPP module
# ------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            *[nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False)
              for r in atrous_rates],
        ])
        self.project = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, 1, bias=False)

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        x = torch.cat(res, dim=1)
        return self.project(x)

# ------------------------------
# 4. DINODeepLab in UNetPP style
# ------------------------------
class DINODeepLab(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, deep_supervision=False,
                 backbone_type="dinov2", backbone_name="dinov2_vitl14"):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Load backbone
        if backbone_type == "dinov2":
            self.backbone, embed_dim = get_dinov2_backbone(backbone_name)
        elif backbone_type == "dinov3":
            self.backbone, embed_dim = get_dinov3_backbone(backbone_name)
        else:
            raise ValueError(f"Unknown backbone_type {backbone_type}")

        # ASPP head
        self.aspp = ASPP(embed_dim, 256)

        # Final classifiers
        if self.deep_supervision:
            self.final = nn.ModuleList([nn.Conv2d(256, num_classes, 1) for _ in range(4)])
        else:
            self.final = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        feats = self.backbone(x)

        # Patch embeddings -> feature map
        if feats.ndim == 3:
            B, N, C = feats.shape
            h = w = int(N ** 0.5)
            feats = feats.transpose(1, 2).reshape(B, C, h, w)
        elif feats.ndim == 2:
            B, C = feats.shape
            feats = feats.view(B, C, 1, 1)

        # ASPP
        feats = self.aspp(feats)

        if self.deep_supervision:
            outputs = [head(feats) for head in self.final]
            outputs = [F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
                       for out in outputs]
            return {"out": outputs[-1], "aux0": outputs[0], "aux1": outputs[1], "aux2": outputs[2]}
        else:
            out = self.final(feats)
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
            return {"out": out}

# ------------------------------
# 5. Test Run
# ------------------------------
if __name__ == "__main__":
    # Example with DINOv2 backbone
    model = DINODeepLab(num_classes=6, backbone_type="dinov2",
                        backbone_name="dinov2_vitb14", deep_supervision=True)

    dummy_input = torch.randn(2, 3, 419, 1024)
    with torch.no_grad():
        out = model(dummy_input)

    for k, v in out.items():
        print(k, v.shape)  # all should be (2, 6, 256, 256)
