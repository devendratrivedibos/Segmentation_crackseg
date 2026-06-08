import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetPP(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        deep_supervision=True,
        base_channels=64,
    ):
        super().__init__()

        nb_filter = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])

        # Bottleneck
        self.conv4_0 = DoubleConv(
            nb_filter[3],
            nb_filter[4],
            dropout=0.3
        )

        # Decoder
        self.conv0_1 = DoubleConv(
            nb_filter[0] + nb_filter[1],
            nb_filter[0]
        )

        self.conv1_1 = DoubleConv(
            nb_filter[1] + nb_filter[2],
            nb_filter[1]
        )

        self.conv2_1 = DoubleConv(
            nb_filter[2] + nb_filter[3],
            nb_filter[2]
        )

        self.conv3_1 = DoubleConv(
            nb_filter[3] + nb_filter[4],
            nb_filter[3]
        )

        self.conv0_2 = DoubleConv(
            nb_filter[0] * 2 + nb_filter[1],
            nb_filter[0]
        )

        self.conv1_2 = DoubleConv(
            nb_filter[1] * 2 + nb_filter[2],
            nb_filter[1]
        )

        self.conv2_2 = DoubleConv(
            nb_filter[2] * 2 + nb_filter[3],
            nb_filter[2]
        )

        self.conv0_3 = DoubleConv(
            nb_filter[0] * 3 + nb_filter[1],
            nb_filter[0]
        )

        self.conv1_3 = DoubleConv(
            nb_filter[1] * 3 + nb_filter[2],
            nb_filter[1]
        )

        self.conv0_4 = DoubleConv(
            nb_filter[0] * 4 + nb_filter[1],
            nb_filter[0]
        )

        if deep_supervision:
            self.final = nn.ModuleList([
                nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
                for _ in range(4)
            ])
        else:
            self.final = nn.Conv2d(
                nb_filter[0],
                num_classes,
                kernel_size=1
            )

        self._initialize_weights()

    def up(self, x, target):
        return F.interpolate(
            x,
            size=target.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(
            torch.cat([x0_0, self.up(x1_0, x0_0)], dim=1)
        )

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(
            torch.cat([x1_0, self.up(x2_0, x1_0)], dim=1)
        )

        x0_2 = self.conv0_2(
            torch.cat([
                x0_0,
                x0_1,
                self.up(x1_1, x0_0)
            ], dim=1)
        )

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(
            torch.cat([x2_0, self.up(x3_0, x2_0)], dim=1)
        )

        x1_2 = self.conv1_2(
            torch.cat([
                x1_0,
                x1_1,
                self.up(x2_1, x1_0)
            ], dim=1)
        )

        x0_3 = self.conv0_3(
            torch.cat([
                x0_0,
                x0_1,
                x0_2,
                self.up(x1_2, x0_0)
            ], dim=1)
        )

        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(
            torch.cat([x3_0, self.up(x4_0, x3_0)], dim=1)
        )

        x2_2 = self.conv2_2(
            torch.cat([
                x2_0,
                x2_1,
                self.up(x3_1, x2_0)
            ], dim=1)
        )

        x1_3 = self.conv1_3(
            torch.cat([
                x1_0,
                x1_1,
                x1_2,
                self.up(x2_2, x1_0)
            ], dim=1)
        )

        x0_4 = self.conv0_4(
            torch.cat([
                x0_0,
                x0_1,
                x0_2,
                x0_3,
                self.up(x1_3, x0_0)
            ], dim=1)
        )

        if self.deep_supervision:

            outputs = [
                self.final[0](x0_1),
                self.final[1](x0_2),
                self.final[2](x0_3),
                self.final[3](x0_4),
            ]

            outputs = [
                F.interpolate(
                    o,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for o in outputs
            ]

            return {
                "out": outputs[-1],
                "aux0": outputs[0],
                "aux1": outputs[1],
                "aux2": outputs[2],
            }

        out = self.final(x0_4)

        out = F.interpolate(
            out,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return {"out": out}

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":

    model = UNetPP(
        in_channels=3,
        num_classes=1,
        deep_supervision=True,
        base_channels=64
    )

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    print(y["out"].shape)

    # torch.Size([2, 1, 256, 256])