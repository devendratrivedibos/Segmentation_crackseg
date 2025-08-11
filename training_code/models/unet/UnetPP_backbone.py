# pip install timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class TimmBackbone(nn.Module):
    def __init__(self, name="efficientnet_b3", pretrained=True, in_chans=3):
        super().__init__()
        self.model = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=in_chans)
        self.out_channels = self.model.feature_info.channels()
        self.reductions = self.model.feature_info.reduction()
    def forward(self, x):
        return self.model(x)  # list of feature maps low->high

class UpCatBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_se=False, use_attn=False):
        super().__init__()
        self.use_attn = use_attn and skip_ch > 0
        if self.use_attn:
            self.attn = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=max(out_ch // 2, 16))
        else:
            self.attn = None
        self.conv1 = ConvBNReLU(in_ch + (skip_ch if skip_ch > 0 else 0), out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
    def forward(self, x_up, x_skip=None):
        if x_skip is not None:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode="bilinear", align_corners=False)
            if self.attn is not None:
                x_skip = self.attn(x_up, x_skip)
            x = torch.cat([x_up, x_skip], dim=1)
        else:
            x = x_up
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x

class SegHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, 1)
    def forward(self, x, out_hw):
        x = self.conv(x)
        if x.shape[2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x

class UNetPP(nn.Module):
    """
    UNet++ with timm encoder, optional SE and attention gates.
    Returns dict:
      - {"out": main_logits, "aux": aux_logits} if deep_supervision=True
      - {"out": main_logits} otherwise
    """
    def __init__(self,
                 encoder_name="efficientnet_b3",
                 encoder_pretrained=True,
                 in_channels=3,
                 num_classes=2,
                 dec_ch=256,
                 use_se=True,
                 use_attn_gates=True,
                 deep_supervision=True):
        super().__init__()
        self.encoder = TimmBackbone(encoder_name, pretrained=encoder_pretrained, in_chans=in_channels)
        chs = self.encoder.out_channels
        if len(chs) >= 5:
            c0, c1, c2, c3, c4 = chs[:5]
            self.use_5 = True
        else:
            c1, c2, c3, c4 = chs[-4:]
            c0 = max(c1 // 2, 16)
            self.use_5 = False
            self.stem = ConvBNReLU(in_channels, c0, k=3, s=2, p=1)

        self.dec_ch = dec_ch
        self.use_se = use_se
        self.use_attn = use_attn_gates
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # projections
        self.proj0 = nn.Conv2d(c0, dec_ch, 1)
        self.proj1 = nn.Conv2d(c1, dec_ch, 1)
        self.proj2 = nn.Conv2d(c2, dec_ch, 1)
        self.proj3 = nn.Conv2d(c3, dec_ch, 1)
        self.proj4 = nn.Conv2d(c4, dec_ch, 1)

        def up(in_ch, skip_ch, out_ch):
            return UpCatBlock(in_ch, skip_ch, out_ch, use_se=use_se, use_attn=use_attn_gates)

        # UNet++ grid
        self.x01 = up(dec_ch, dec_ch, dec_ch)
        self.x11 = up(dec_ch, dec_ch, dec_ch)
        self.x21 = up(dec_ch, dec_ch, dec_ch)
        self.x31 = up(dec_ch, dec_ch, dec_ch)

        self.x02 = up(dec_ch, dec_ch * 2, dec_ch)
        self.x12 = up(dec_ch, dec_ch * 2, dec_ch)
        self.x22 = up(dec_ch, dec_ch * 2, dec_ch)

        self.x03 = up(dec_ch, dec_ch * 3, dec_ch)
        self.x13 = up(dec_ch, dec_ch * 3, dec_ch)

        self.x04 = up(dec_ch, dec_ch * 4, dec_ch)

        # heads
        self.head_main = SegHead(dec_ch, num_classes)  # from X_{0,4}
        if self.deep_supervision:
            self.head_aux = SegHead(dec_ch, num_classes)   # from X_{0,3}

    def forward(self, x):
        # Correct spatial dims
        # H, W = x.shape[2], x.shape[1]
        H, W = x.shape[2], x.shape[3]  # ✅
        feats = self.encoder(x)
        if self.use_5:
            e0, e1, e2, e3, e4 = feats[:5]
        else:
            e1, e2, e3, e4 = feats[-4:]
            e0 = F.interpolate(self.stem(x), size=e1.shape[2:], mode="bilinear", align_corners=False)

        x00 = self.proj0(e0)
        x10 = self.proj1(e1)
        x20 = self.proj2(e2)
        x30 = self.proj3(e3)
        x40 = self.proj4(e4)

        # column j=1
        x01 = self.x01(x10, x00)
        x11 = self.x11(x20, x10)
        x21 = self.x21(x30, x20)
        x31 = self.x31(x40, x30)

        # column j=2
        x02 = self.x02(x11, torch.cat([x00, x01], dim=1))
        x12 = self.x12(x21, torch.cat([x10, x11], dim=1))
        x22 = self.x22(x31, torch.cat([x20, x21], dim=1))

        # column j=3
        x03 = self.x03(x12, torch.cat([x00, x01, x02], dim=1))
        x13 = self.x13(x22, torch.cat([x10, x11, x12], dim=1))

        # column j=4
        x04 = self.x04(x13, torch.cat([x00, x01, x02, x03], dim=1))

        out = self.head_main(x04, (H, W))
        if self.deep_supervision:
            aux = self.head_aux(x03, (H, W))
            return {"out": out, "aux": aux}
        else:
            return {"out": out}

def build_unetpp_model(
    encoder="efficientnet_b3",   # or "resnet50"
    pretrained=True,
    in_channels=3,
    num_classes=2,
    dec_ch=256,
    use_se=True,
    use_attn_gates=True,
    deep_supervision=True
):
    return UNetPP(
        encoder_name=encoder,
        encoder_pretrained=pretrained,
        in_channels=in_channels,
        num_classes=num_classes,
        dec_ch=dec_ch,
        use_se=use_se,
        use_attn_gates=use_attn_gates,
        deep_supervision=deep_supervision
    )
