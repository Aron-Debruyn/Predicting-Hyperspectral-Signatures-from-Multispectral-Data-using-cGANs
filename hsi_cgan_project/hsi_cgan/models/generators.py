"""Generator networks (the "G" in the conditional GANs).

Each generator maps a low-dimensional multispectral input to a full hyperspectral output.

Contents
--------
CNNGenerator              Plain 5-layer 3x3 conv network (64-128-256-512-out) with ReLU, no normalisation.
UNetGenerator             Encoder-decoder with skip connections (5 down blocks, 4 up blocks, tanh output);
                          pads upsampled maps to match the skip connections.
ResidualBlock             Building block of ResNetGenerator: 2x (conv + InstanceNorm + PReLU) with an optional
                          1x1 skip projection; the residual is scaled by 0.3.
ResNetGenerator           5x5 conv expansion (1024 ch), compression (256 ch), two ResidualBlocks, expansion
                          again, plus a 7x7 conv skip path from the input.
GeneratorLarge            Pixel-wise (spectral) generator: three Conv1d layers + 4-layer fully connected head
                          with BatchNorm/Dropout, sigmoid output. FC input size 3*10.
GeneratorLarge_MI_PCA     Same as GeneratorLarge but FC input size 5*10 (used for the 20-band MI / PCA inputs).
GeneratorLarge_bandwidth  Same as GeneratorLarge but FC input size 9*10 (used for the 56-band "bandwidth" input).
GeneratorMLP              Pixel-wise 4-layer MLP (128-256-512-out) with Dropout(0.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Generator class
# ========================= GENERATORS =========================
class CNNGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=9):
        super(CNNGenerator, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, output_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetGenerator, self).__init__()

        def down_block(in_c, out_c, apply_batchnorm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False)]
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_c, out_c, apply_dropout=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU(inplace=True)]
            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # encoder
        self.down1 = down_block(input_channels, 64, apply_batchnorm=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)

        # Decoder
        self.up4 = up_block(512, 512, apply_dropout=True)
        self.up3 = up_block(1024, 256, apply_dropout=True)
        self.up2 = up_block(512, 128, apply_dropout=True)
        self.up1 = up_block(256, 64)


        self.final_layer = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1) # uit is 128
        d3 = self.down3(d2) # uit is 256
        d4 = self.down4(d3) # uit is 512
        d5 = self.down5(d4) # uit is 512

        u4 = self.up4(d5)

        diffY = d4.size()[2] - u4.size()[2]
        diffX = d4.size()[3] - u4.size()[3]

        u4 = F.pad(u4, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        u4 = torch.cat([u4, d4], dim=1)
        u3 = self.up3(u4)

        diffY = d3.size()[2] - u3.size()[2]
        diffX = d3.size()[3] - u3.size()[3]

        u3 = F.pad(u3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        u3 = torch.cat([u3, d3], dim=1)
        u2 = self.up2(u3)

        diffY = d2.size()[2] - u2.size()[2]
        diffX = d2.size()[3] - u2.size()[3]

        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        u2 = torch.cat([u2, d2], dim=1)
        u1 = self.up1(u2)



        diffY = d1.size()[2] - u1.size()[2]
        diffX = d1.size()[3] - u1.size()[3]


        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        u1 = torch.cat([u1, d1], dim=1)

        return torch.tanh(self.final_layer(u1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size//2)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size//2)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.prelu2 = nn.PReLU()

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.skip = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.prelu2(out)

        if self.skip is not None:
            residual = self.skip(residual)

        out += residual  * 0.3
        return out


class ResNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=9):
        super(ResNetGenerator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 1024, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        self.compress_conv = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        self.res_block1 = ResidualBlock(256, 256, kernel_size=3)
        self.res_block2 = ResidualBlock(256, 256, kernel_size=3)

        self.skip_path = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=1, padding=3),
            nn.PReLU()
        )

        self.expand_conv1 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.expand_conv2 = nn.Sequential(
            nn.Conv2d(1024, output_channels, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

    def forward(self, x):
        out_initial = self.initial_conv(x)
        out_compress = self.compress_conv(out_initial)
        out_main1 = self.res_block1(out_compress)
        out_main2 = self.res_block2(out_main1)

        out_skip = self.skip_path(x)

        out1 = self.expand_conv1(out_main2 + out_compress)

        if out1.size() != out_initial.size():
            diffY = out_initial.size()[2] - out1.size()[2]
            diffX = out_initial.size()[3] - out1.size()[3]
            out1 = F.pad(out1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

        out2 = self.expand_conv2(out1 + out_initial)

        if out2.size() != out_skip.size():
            diffY = out_skip.size()[2] - out2.size()[2]
            diffX = out_skip.size()[3] - out2.size()[3]
            out2 = F.pad(out2, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

        return out2 + out_skip


class GeneratorLarge(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorLarge, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=50, out_channels=10, kernel_size=3, stride=2, padding=2)

        self.fc = nn.Sequential(
            nn.Linear(3 * 10, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(4096, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(2048, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)


        x = x.view(x.size(0), -1)
        hyperspectral_data = self.fc(x)



        return hyperspectral_data


class GeneratorLarge_MI_PCA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorLarge_MI_PCA, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=50, out_channels=10, kernel_size=3, stride=2, padding=2)

        self.fc = nn.Sequential(
            nn.Linear(5 * 10, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(4096, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(2048, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)


        x = x.view(x.size(0), -1)
        hyperspectral_data = self.fc(x)



        return hyperspectral_data


class GeneratorLarge_bandwidth(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorLarge_bandwidth, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=50, out_channels=10, kernel_size=3, stride=2, padding=2)

        self.fc = nn.Sequential(
            nn.Linear(9 * 10, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(4096, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048, momentum=0.4),
            nn.Dropout(0.5),

            nn.Linear(2048, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)


        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)


        x = x.view(x.size(0), -1)
        hyperspectral_data = self.fc(x)



        return hyperspectral_data


# Generator Network (MLP)
class GeneratorMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)
