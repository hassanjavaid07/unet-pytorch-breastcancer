import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint



# ========================= START OF UNET FUNCTIONS ==================================

# Implements double conv layer consisting of Conv23 => BatchNorm => ReLU
class InDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.in_conv(x)

# Implements encoder stage downscaling and then double conv to encode input
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            InDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Implements decoder stage upscaling and then double conv the encoded input
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = InDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Implements final output conv layer in UNet
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# U-net model class implementation
class UNetModel(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder layers
        self.encoder = nn.Sequential(
            InDoubleConv(self.n_channels, 64),    # (convolution => [BN] => ReLU) * 2
            Down(64, 128),                    # Downscaling => Maxpool => InDoubleConv
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            Up(1024, 512),                   # Upscaling => Upsample => InDoubleConv
            Up(512, 256),
            Up(256, 128),
            Up(128, 64)
        )

        self.output = OutConv(64, self.n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder Stage
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)

        # Decoder Stage
        x = self.decoder[0](x5, x4)                # Skip connections implemented
        x = self.decoder[1](x, x3)
        x = self.decoder[2](x, x2)
        x = self.decoder[3](x, x1)
        out_logits = self.output(x)
        final = self.sigmoid(out_logits)    # Final sigmoid for limiting out-range

        return final

    def use_checkpointing(self):
        self.encoder = torch.utils.checkpoint(self.encoder)
        self.decoder = torch.utils.checkpoint(self.decoder)


# ========================= END OF UNET FUNCTIONS ==================================
