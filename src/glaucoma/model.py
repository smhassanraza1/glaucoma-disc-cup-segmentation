"""U-Net with a pretrained ResNet34 encoder.

A U-Net is an encoder that shrinks the image step by step to capture context ("this is an optic disc"),
and a decoder that grows it back to full size to say exactly which pixels. Skip connections copy the
encoder's detailed feature maps across to the decoder, so boundaries stay sharp.

The encoder is a ResNet34 already trained on ImageNet. It starts out knowing edges and textures, which
matters with only a few hundred training photos.

Output channels are independent yes/no maps rather than one multi-way choice, because the cup lies inside
the disc: a cup pixel is also a disc pixel. The disc locator uses one channel (disc), the segmenter two
(disc, cup).
"""

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Double the resolution, join the matching encoder features, then convolve."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Input ``(B, 3, H, W)`` with H and W divisible by 32. Output logits ``(B, n_classes, H, W)``."""

    def __init__(self, pretrained: bool = True, n_classes: int = 2):
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 64 ch, 1/2
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2  # 64 ch 1/4, 128 ch 1/8
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4  # 256 ch 1/16, 512 ch 1/32

        self.up4 = UpBlock(512, 256, 256)  # 1/16
        self.up3 = UpBlock(256, 128, 128)  # 1/8
        self.up2 = UpBlock(128, 64, 64)  # 1/4
        self.up1 = UpBlock(64, 64, 32)  # 1/2
        self.up0 = UpBlock(32, 0, 16)  # full size, no skip at this level
        self.head = nn.Conv2d(16, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        d = self.up4(s5, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, None)
        return self.head(d)
