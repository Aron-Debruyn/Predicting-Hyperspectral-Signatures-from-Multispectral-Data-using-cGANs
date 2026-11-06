"""Factories that build the training components from the names stored in ``hyperparameters``.

Contents
--------
creating_generator_and_discriminator  Build (generator, discriminator) from their names (training version).
creating_loss_function                Look up a torch loss module by name (MSE, L1, BCE, BCEWithLogitsLoss, ...).
creating_optimizer                    Build the generator and discriminator optimisers (Adam, SGD, RMSprop, ...).

Note: a second variant of ``creating_generator_and_discriminator`` that also selects the
GeneratorLarge_MI_PCA / GeneratorLarge_bandwidth classes lives in ``evaluation/checkpoint_loading.py``.
"""

import torch.nn as nn
import torch.optim as optim
from hsi_cgan.models.generators import CNNGenerator, UNetGenerator, ResNetGenerator, GeneratorLarge, GeneratorMLP
from hsi_cgan.models.discriminators import DiscriminatorMLP, PatchGANDiscriminator, DiscriminatorLarge


def creating_generator_and_discriminator(name_g, name_d, input_channels, output_channels, hard_code_known):
  if name_g == 'CNNGenerator':
    generator = CNNGenerator(input_channels, output_channels)
  elif name_g == 'UNetGenerator':
    generator = UNetGenerator(input_channels, output_channels)
  elif name_g == 'ResNetGenerator':
    generator = ResNetGenerator(input_channels, output_channels)
  elif name_g == 'GeneratorLarge':
    generator = GeneratorLarge(input_channels, output_channels)
  elif name_g == 'MLP':
    generator = GeneratorMLP(input_channels, output_channels)


  if name_d == 'PatchGANDiscriminator':
    discriminator = PatchGANDiscriminator(input_channels, output_channels)
  elif name_d == 'DiscriminatorLarge':
    discriminator = DiscriminatorLarge(input_channels, output_channels)
  elif name_d == 'MLP':
    discriminator = DiscriminatorMLP(input_channels, output_channels)

  return generator, discriminator


def creating_loss_function(name: str):
    loss_dict = {
        'MSE': nn.MSELoss(),
        'L1': nn.L1Loss(),
        'BCE': nn.BCELoss(),
        'CrossEntropy': nn.CrossEntropyLoss(),
        'SmoothL1Loss': nn.SmoothL1Loss(),
        'KLDivLoss': nn.KLDivLoss(),
        'HingeEmbeddingLoss': nn.HingeEmbeddingLoss(),
        'MarginRankingLoss': nn.MarginRankingLoss(),
        'MultiLabelSoftMarginLoss': nn.MultiLabelSoftMarginLoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss()
    }


    criterion = loss_dict.get(name)

    if criterion is None:
        raise ValueError(f"Unknown loss function name: {name}")

    return criterion


def creating_optimizer(name_optim, generator, discriminator, learning_rate:float, betas:tuple):
  if name_optim == 'Adam':
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
  elif name_optim == 'AdamDecay':
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-5)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)
  elif name_optim == 'SGD':
    g_optimizer = optim.SGD(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.SGD(discriminator.parameters(), lr=learning_rate)
  elif name_optim == 'RMSprop':
    g_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=learning_rate)

  elif name_optim == 'Adagrad':
    g_optimizer = optim.Adagrad(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adagrad(discriminator.parameters(), lr=learning_rate)

  elif name_optim == 'Adadelta':
    g_optimizer = optim.Adadelta(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adadelta(discriminator.parameters(), lr=learning_rate)
  return g_optimizer, d_optimizer
