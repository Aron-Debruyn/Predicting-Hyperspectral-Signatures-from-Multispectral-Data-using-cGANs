"""Discriminator networks (the "D" in the conditional GANs).

Each discriminator receives the multispectral input and a (real or generated) hyperspectral output.

Contents
--------
DiscriminatorMLP         Pixel-wise MLP discriminator on the concatenated input/output vectors (sigmoid output).
PatchGANDiscriminator    Spectral-normalised conv discriminator with a patch output (usable as an error map via
                         ``error_map=True``) and a global output.
DiscriminatorLarge       Spectral-normalised fully connected discriminator for the pixel-wise (Cramer-style) setup.
"""

import torch
import torch.nn as nn


# ========================= DISCRIMINATORS =========================
# Discriminator Network (MLP)
class DiscriminatorMLP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DiscriminatorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels+output_channels, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_image, multispectral_image):
        x = torch.cat((rgb_image, multispectral_image), dim=1)
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(PatchGANDiscriminator, self).__init__()

        self.features = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels + output_channels, input_channels + output_channels, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(input_channels + output_channels, 512, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )


        self.patch_output = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=3, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),
        )


        self.global_output = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 256, kernel_size=15, stride=1, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, kernel_size=1, stride=1)),
        )

    def forward(self, rgb_image, multispectral_image, error_map=False):
        x = torch.cat((rgb_image, multispectral_image), dim=1)
        features = self.features(x)

        if error_map:
          patch_out = self.patch_output(features)
          global_out = self.global_output(features)
          return patch_out, global_out
        else:

          global_out = self.global_output(features)
          return global_out


class DiscriminatorLarge(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(DiscriminatorLarge, self).__init__()
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_channels + output_channels, 1024)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 1024)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 256)),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, rgb_image, multispectral_image):
        x = torch.cat((rgb_image, multispectral_image), dim=1)
        validity = self.fc(x)
        return validity
