"""Validation used inside the training loop.

Contents
--------
validate  Average RMSE and MRAE of a generator over a dataloader (training version: no SAM/PSNR/SSIM).
          The richer evaluation version lives in ``evaluation/validators.py``.
"""

import torch
from hsi_cgan.metrics.losses import rmse_loss, mrae_loss


def validate(generator, val_dataloader, device, indices, hard_code_known):
    generator.eval()
    total_rmse = 0.0
    total_mrae = 0.0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for ms_images, hs_images in val_dataloader:

            ms_images = ms_images.to(device)
            hs_images = hs_images.to(device)


            if hard_code_known:
                fake_hyperspectral_images = generator(ms_images, hs_images, indices)
            else:
                fake_hyperspectral_images = generator(ms_images)


            batch_rmse = rmse_loss(fake_hyperspectral_images, hs_images)
            batch_mrae = mrae_loss(fake_hyperspectral_images, hs_images)
            total_rmse += batch_rmse.item()
            total_mrae += batch_mrae.item()


    avg_rmse = total_rmse / num_batches
    avg_mrae = total_mrae / num_batches
    return avg_rmse, avg_mrae
