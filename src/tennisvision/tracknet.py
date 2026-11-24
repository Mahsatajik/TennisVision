"""TrackNet V1 architecture compatible with the separately attributed tennis weights.

Network structure: Huang et al., AVSS 2019; checkpoint key layout follows the
yastrebksv/TrackNet PyTorch implementation. Checkpoints load with weights_only=True.
"""

from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [
            9,
            64,
            64,
            128,
            128,
            256,
            256,
            256,
            512,
            512,
            512,
            256,
            256,
            256,
            128,
            128,
            64,
            64,
            256,
        ]
        for i, (a, b) in enumerate(zip(channels, channels[1:]), 1):
            setattr(self, f"conv{i}", ConvBlock(a, b))
        for i in (1, 2, 3):
            setattr(self, f"pool{i}", nn.MaxPool2d(2, 2))
            setattr(self, f"ups{i}", nn.Upsample(scale_factor=2))

    def forward(self, x):
        for i in range(1, 19):
            x = getattr(self, f"conv{i}")(x)
            if i in (2, 4, 7):
                x = getattr(self, f"pool{(2, 4, 7).index(i) + 1}")(x)
            if i in (10, 13, 15):
                x = getattr(self, f"ups{(10, 13, 15).index(i) + 1}")(x)
        return x.reshape(x.shape[0], 256, -1)
