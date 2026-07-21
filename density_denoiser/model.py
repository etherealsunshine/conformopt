from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DensityDenoiser(nn.Module):
    """Shape-preserving 3D U-Net for 32^3 crystallographic patches."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool3d(2)
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)
        self.enc4 = ConvBlock(4 * c, 8 * c)
        self.bottleneck = ConvBlock(8 * c, 8 * c)

        self.up4 = nn.ConvTranspose3d(8 * c, 8 * c, 2, stride=2)
        self.dec4 = ConvBlock(16 * c, 4 * c)
        self.up3 = nn.ConvTranspose3d(4 * c, 4 * c, 2, stride=2)
        self.dec3 = ConvBlock(8 * c, 2 * c)
        self.up2 = nn.ConvTranspose3d(2 * c, 2 * c, 2, stride=2)
        self.dec2 = ConvBlock(4 * c, c)
        # The original prompt omitted this 16^3 -> 32^3 stage.
        self.up1 = nn.ConvTranspose3d(c, c, 2, stride=2)
        self.dec1 = ConvBlock(2 * c, c)
        self.final = nn.Conv3d(c, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        middle = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat((self.up4(middle), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.final(d1)


class ResidualDensityDenoiser(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.unet = DensityDenoiser(base_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.unet(inputs)


def spatial_gradient(volume: torch.Tensor) -> torch.Tensor:
    dx = torch.diff(volume, dim=2, append=volume[:, :, -1:, :, :])
    dy = torch.diff(volume, dim=3, append=volume[:, :, :, -1:, :])
    dz = torch.diff(volume, dim=4, append=volume[:, :, :, :, -1:])
    return torch.cat((dx, dy, dz), dim=1)
