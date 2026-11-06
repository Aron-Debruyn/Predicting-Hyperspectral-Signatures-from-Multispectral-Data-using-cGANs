"""Saving and loading of training state.

Contents
--------
save_checkpoint        Save generator/discriminator/optimiser states, loss histories and epoch to disk.
load_checkpoint        Restore a checkpoint written by save_checkpoint (training version; expects the
                       ``val_losses_rmse`` / ``val_losses_mae`` keys).
save_hyperparameters   Dump the hyperparameters dict to JSON (numpy values converted to plain Python).

Note: an older variant of ``load_checkpoint`` (key ``val_losses``) lives in
``evaluation/checkpoint_loading.py``.
"""

import json
import numpy as np
import torch


def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, g_losses, d_losses, val_losses_rmse, val_losses_mae, filename):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
        'val_losses_rmse': val_losses_rmse,
        'val_losses_mae': val_losses_mae
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, filename, hyperparameters):
    checkpoint = torch.load("{0}/checkpoints/".format(hyperparameters['basedir'])+filename,  map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    g_losses = checkpoint['g_losses']
    d_losses = checkpoint['d_losses']
    val_losses_rmse = checkpoint['val_losses_rmse']
    # val_losses_mae = []
    val_losses_mae = checkpoint['val_losses_mae']
    print(f"Checkpoint loaded from {filename}")
    return epoch, g_losses, d_losses, val_losses_rmse, val_losses_mae


def save_hyperparameters(hyperparameters, filename):

    modified_hyperparameters = hyperparameters.copy()


    for key, value in modified_hyperparameters.items():
        if isinstance(value, (np.int64, np.int32)):
            modified_hyperparameters[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            modified_hyperparameters[key] = float(value)
        elif isinstance(value, np.ndarray):
            modified_hyperparameters[key] = value.tolist()


    with open(filename, "w") as f:
        json.dump(modified_hyperparameters, f, indent=4)
