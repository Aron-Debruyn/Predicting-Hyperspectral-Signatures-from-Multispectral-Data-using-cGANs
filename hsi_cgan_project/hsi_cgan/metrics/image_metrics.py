"""Image / spectral quality metrics used during evaluation.

Contents
--------
sam_loss                       Spectral Angle Mapper (mean angle between predicted and target spectra).
calculate_metrics              RMSE, MRAE, SAM, PSNR and SSIM for a batch.
calculate_per_channel_metrics  The same five metrics computed separately for every spectral band.
"""

import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


def sam_loss(pred, target, eps=1e-8):
    pred = pred.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
    target = target.permute(0, 2, 3, 1).contiguous()

    dot_product = torch.sum(pred * target, dim=-1)
    norm_pred = torch.norm(pred, p=2, dim=-1)
    norm_target = torch.norm(target, p=2, dim=-1)

    cosine_sim = dot_product / (norm_pred * norm_target + eps)
    cosine_sim = torch.clamp(cosine_sim, -1.0 + eps, 1.0 - eps)
    return torch.mean(torch.acos(cosine_sim))


def calculate_metrics(pred, target, eps=1e-8):
    results = {}

    # RMSE
    results['RMSE'] = torch.sqrt(F.mse_loss(pred, target))

    # MRAE
    relative_errors = torch.abs(target - pred) / (torch.abs(target) + eps)
    results['MRAE'] = torch.mean(relative_errors)


    pred_norm = F.normalize(pred.permute(0,2,3,1), p=2, dim=-1)
    target_norm = F.normalize(target.permute(0,2,3,1), p=2, dim=-1)
    cos_sim = torch.sum(pred_norm * target_norm, dim=-1).flatten()
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    results['SAM'] = torch.mean(torch.acos(cos_sim))


    results['PSNR'] = psnr(pred, target)
    results['SSIM'] = ssim(pred, target)

    return results


def calculate_per_channel_metrics(pred, target, eps=1e-8):
    metrics = {
        'RMSE': [], 'MRAE': [], 'SAM': [], 'PSNR': [], 'SSIM': []
    }
    num_channels = pred.shape[1]

    for c in range(num_channels):
        pred_ch = pred[:, c:c+1, :, :]
        target_ch = target[:, c:c+1, :, :]

        # RMSE
        metrics['RMSE'].append(torch.sqrt(F.mse_loss(pred_ch, target_ch)))
        # MRAE
        relative_errors = torch.abs(target_ch - pred_ch) / (torch.abs(target_ch) + eps)
        metrics['MRAE'].append(torch.mean(relative_errors))
        # SAM
        pred_norm = F.normalize(pred_ch.permute(0,2,3,1), p=2, dim=-1)
        target_norm = F.normalize(target_ch.permute(0,2,3,1), p=2, dim=-1)
        cos_sim = torch.sum(pred_norm * target_norm, dim=-1).flatten()
        cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
        metrics['SAM'].append(torch.mean(torch.acos(cos_sim)))
        # PSNR
        metrics['PSNR'].append(psnr(pred_ch, target_ch))
        # SSIM
        metrics['SSIM'].append(ssim(pred_ch, target_ch))

    return {k: torch.stack(v) for k, v in metrics.items()}
