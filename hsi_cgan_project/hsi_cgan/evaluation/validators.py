"""Evaluation routines that run a trained generator over a dataloader.

Contents
--------
validate               Average RMSE, MRAE, SAM, PSNR and SSIM (evaluation version; handles the pixel-wise
                       'MLP' architecture by reshaping the patches).
validate_extended      Pixel and spectral RMSE/MRAE plus bootstrapped confidence intervals; plots the mean
                       spectral curves.
validate_per_channel   All five metrics averaged overall and per spectral band.
"""

import numpy as np
import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from hsi_cgan.metrics.losses import rmse_loss, mrae_loss
from hsi_cgan.metrics.image_metrics import sam_loss, calculate_metrics, calculate_per_channel_metrics
from hsi_cgan.visualization.spectral_plots import plot_spectral_curves


def validate(generator, val_dataloader, device, architecture, bands):
    generator.eval()
    total_rmse, total_mrae, total_sam = 0.0, 0.0, 0.0
    total_psnr, total_ssim = 0.0, 0.0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for ms_images, hs_images in val_dataloader:
            ms_images = ms_images.to(device)
            hs_images = hs_images.to(device)


            if architecture == 'MLP':

                B, C, H, W = ms_images.shape

                ms_images = ms_images.permute(2, 3, 0, 1).reshape(-1, C)


                fake_hs_images = generator(ms_images)


                C_out = fake_hs_images.shape[-1]
                fake_hs_images = fake_hs_images.reshape(H, W, B, C_out).permute(2, 3, 0, 1)
            else:
                fake_hs_images = generator(ms_images)

            batch_rmse = rmse_loss(fake_hs_images, hs_images)
            batch_mrae = mrae_loss(fake_hs_images, hs_images)
            batch_sam = sam_loss(fake_hs_images, hs_images)
            batch_psnr = psnr(fake_hs_images, hs_images)
            batch_ssim = ssim(fake_hs_images, hs_images)


            total_rmse += batch_rmse.item()
            total_mrae += batch_mrae.item()
            total_sam += batch_sam.item()
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()


    avg_rmse = total_rmse / num_batches
    avg_mrae = total_mrae / num_batches
    avg_sam = total_sam / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    return {
        'RMSE': avg_rmse,
        'MRAE': avg_mrae,
        'SAM': avg_sam,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim
    }


def validate_extended(generator, val_dloader, device, architecture, indices, n_bootstraps=1000, confidence=0.95):
    generator.eval()
    total_rmse, total_mae = 0.0, 0.0
    num_batches = len(val_dloader)


    true_spectra = []
    pred_spectra = []

    with torch.no_grad():
        for ms_images, hs_images in val_dloader:
            ms_images = ms_images.to(device)
            hs_images = hs_images.to(device)

            if architecture == 'MLP':

                B, C, H, W = ms_images.shape

                ms_images = ms_images.permute(2, 3, 0, 1).reshape(-1, C)


                fake_hs_images = generator(ms_images)


                C_out = fake_hs_images.shape[-1]
                fake_hs_images = fake_hs_images.reshape(H, W, B, C_out).permute(2, 3, 0, 1)
            else:
                fake_hs_images = generator(ms_images)


            total_rmse += rmse_loss(fake_hs_images, hs_images).item()
            total_mae += mrae_loss(fake_hs_images, hs_images).item()


            true_spectra.append(hs_images.mean(dim=(0, 2, 3)).cpu().numpy())
            pred_spectra.append(fake_hs_images.mean(dim=(0, 2, 3)).cpu().numpy())


    true_spectra = np.stack(true_spectra)
    pred_spectra = np.stack(pred_spectra)


    true_mean = true_spectra.mean(axis=0)
    pred_mean = pred_spectra.mean(axis=0)


    def bootstrap_ci(data, n_bootstraps=1000, confidence=0.95):
        n_samples = data.shape[0]
        bootstrap_means = np.zeros((n_bootstraps, data.shape[1]))

        for i in range(n_bootstraps):

            resampled = data[np.random.choice(n_samples, size=n_samples, replace=True)]
            bootstrap_means[i] = resampled.mean(axis=0)


        alpha = (1 - confidence) / 2
        ci_low = np.percentile(bootstrap_means, 100 * alpha, axis=0)
        ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha), axis=0)
        return ci_low, ci_high

    true_ci_low, true_ci_high = bootstrap_ci(true_spectra, n_bootstraps, confidence)
    pred_ci_low, pred_ci_high = bootstrap_ci(pred_spectra, n_bootstraps, confidence)

    epsilon=1e-7

    spectral_rmse = np.sqrt(np.mean((true_mean - pred_mean) ** 2))
    spectral_mrae = np.mean(np.abs(true_mean - pred_mean) / (np.abs(true_mean) + epsilon))




    plot_spectral_curves(true_mean, pred_mean, true_ci_low, true_ci_high, pred_ci_low, pred_ci_high, indices)

    return {
        "pixel_rmse": total_rmse / num_batches,
        "pixel_mrae": total_mae / num_batches,
        "spectral_rmse": spectral_rmse,
        "spectral_mrae": spectral_mrae,
        "true_ci": (true_ci_low, true_ci_high),
        "pred_ci": (pred_ci_low, pred_ci_high)
    }


def validate_per_channel(generator, val_dataloader, device, architecture, indices, eps=1e-8):
    generator.eval()
    metric_sums = {
        'RMSE': torch.zeros(1, device=device),
        'MRAE': torch.zeros(1, device=device),
        'SAM': torch.zeros(1, device=device),
        'PSNR': torch.zeros(1, device=device),
        'SSIM': torch.zeros(1, device=device),
        'RMSE_per_channel': None, 'MRAE_per_channel': None, 'SAM_per_channel': None,
        'PSNR_per_channel': None, 'SSIM_per_channel': None
    }
    num_batches = len(val_dataloader)

    sum_per_channel, sum_sq_per_channel = None, None
    num_pixels_total = 0

    with torch.no_grad():
        for ms_images, hs_images in val_dataloader:
            ms_images = ms_images.to(device)
            hs_images = hs_images.to(device)

            # --- Variance Calculation Step ---
            if sum_per_channel is None:
                num_channels = hs_images.shape[1]
                sum_per_channel = torch.zeros(num_channels, device=device, dtype=torch.float64)
                sum_sq_per_channel = torch.zeros(num_channels, device=device, dtype=torch.float64)

            num_pixels_in_batch = hs_images.shape[0] * hs_images.shape[2] * hs_images.shape[3]
            num_pixels_total += num_pixels_in_batch
            sum_per_channel += torch.sum(hs_images, dim=[0, 2, 3], dtype=torch.float64)
            sum_sq_per_channel += torch.sum(hs_images**2, dim=[0, 2, 3], dtype=torch.float64)
            # --- End Variance Calculation Step ---

            if architecture == 'MLP':
                B, C, H, W = ms_images.shape
                ms_images = ms_images.permute(2, 3, 0, 1).reshape(-1, C)
                fake_hs_images = generator(ms_images)
                C_out = fake_hs_images.shape[-1]
                fake_hs_images = fake_hs_images.reshape(H, W, B, C_out).permute(2, 3, 0, 1)
            else:
                fake_hs_images = generator(ms_images)

            batch_metrics = calculate_metrics(fake_hs_images, hs_images)
            channel_metrics = calculate_per_channel_metrics(fake_hs_images, hs_images)

            if metric_sums['RMSE_per_channel'] is None:
                num_channels_pred = fake_hs_images.shape[1]
                for k in ['RMSE', 'MRAE', 'SAM', 'PSNR', 'SSIM']:
                    metric_sums[f'{k}_per_channel'] = torch.zeros(num_channels_pred, device=device)

            for k in ['RMSE', 'MRAE', 'SAM', 'PSNR', 'SSIM']:
                metric_sums[k] += batch_metrics[k]
                metric_sums[f'{k}_per_channel'] += channel_metrics[k]

    mean = sum_per_channel / num_pixels_total
    mean_sq = sum_sq_per_channel / num_pixels_total
    variance_per_channel = mean_sq - mean**2

    results = {}
    for k in ['RMSE', 'MRAE', 'SAM', 'PSNR', 'SSIM']:
        results[k] = (metric_sums[k] / num_batches).cpu().item()

        avg_per_channel_metric = metric_sums[f'{k}_per_channel'] / num_batches


        # variance_on_device = variance_per_channel.to(avg_per_channel_metric.device, dtype=avg_per_channel_metric.dtype)


        normalized_metric = avg_per_channel_metric #/ (variance_on_device + eps) #####VARIANCE OR NO VARIANCE

        results[f'{k}_per_channel'] = normalized_metric.cpu().tolist()

    return results
