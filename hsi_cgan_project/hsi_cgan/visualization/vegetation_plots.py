"""Real vs. predicted comparison plots (RGB and vegetation indices).

Contents
--------
plot_vegetation_index_comparison  Side-by-side RGB and index maps (NDVI, CARI, DD, REIP, Clred_edge, REP_LE) for
                                  the real and predicted cube, printing the SSIM of each index.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity
from hsi_cgan.evaluation.vegetation_indices import calculate_vegetation_index


def plot_vegetation_index_comparison(real_hsi, predicted_hsi, wavelengths, index_names=None):
    if index_names is None:
        index_names = ['NDVI', 'CARI', 'DD', 'REIP', 'Clred_edge', 'REP_LE']

    # --- RGB Plot ---
    try:
        # Find RGB wavelength indices (approximate)
        red_idx = np.argmin(np.abs(wavelengths - 650))   # Red ~650nm
        green_idx = np.argmin(np.abs(wavelengths - 550)) # Green ~550nm
        blue_idx = np.argmin(np.abs(wavelengths - 450))  # Blue ~450nm

        # Extract RGB bands and normalize
        real_rgb = real_hsi[:, :, [red_idx, green_idx, blue_idx]]
        pred_rgb = predicted_hsi[:, :, [red_idx, green_idx, blue_idx]]

        # Normalize to [0,1] for display
        real_rgb_norm = (real_rgb - np.min(real_rgb)) / (np.max(real_rgb) - np.min(real_rgb))
        pred_rgb_norm = (pred_rgb - np.min(pred_rgb)) / (np.max(pred_rgb) - np.min(pred_rgb))

        fig_rgb = plt.figure(figsize=(12.5, 6))
        gs_rgb = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

        ax1_rgb = fig_rgb.add_subplot(gs_rgb[0, 0])
        ax2_rgb = fig_rgb.add_subplot(gs_rgb[0, 1])

        fig_rgb.suptitle('RGB Comparison', fontsize='xx-large')

        ax1_rgb.imshow(real_rgb_norm)
        ax1_rgb.set_title('Real Hyperspectral', fontsize='xx-large')
        ax1_rgb.set_xticks([])
        ax1_rgb.set_yticks([])

        ax2_rgb.imshow(pred_rgb_norm)
        ax2_rgb.set_title('Predicted Hyperspectral', fontsize='xx-large')
        ax2_rgb.set_xticks([])
        ax2_rgb.set_yticks([])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Failed to plot RGB: {str(e)}")

    # --- Vegetation Index Plots ---
    for index_name in index_names:
        try:
            real_index = calculate_vegetation_index(index_name, real_hsi, wavelengths)
            pred_index = calculate_vegetation_index(index_name, predicted_hsi, wavelengths)

            # --- SSIM calculation ---
            real_index_nonan = np.nan_to_num(real_index)
            pred_index_nonan = np.nan_to_num(pred_index)

            data_range = max(np.nanmax(real_index_nonan), np.nanmax(pred_index_nonan)) - min(np.nanmin(real_index_nonan), np.nanmin(pred_index_nonan))
            if data_range <= 0:
                data_range = 1.0

            ssim_score = structural_similarity(real_index_nonan, pred_index_nonan, data_range=data_range)
            print(f"SSIM for {index_name}: {ssim_score}")
            # --- end ---

            fig = plt.figure(figsize=(12.5, 6))
            gs = gridspec.GridSpec(1, 3, width_ratios=[20, 20, 1.5], wspace=0.25)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            cax = fig.add_subplot(gs[0, 2])

            fig.suptitle(f'{index_name} Vegetation Index Comparison', fontsize='xx-large')

            if index_name == 'NDVI':
                cmap = 'RdYlGn'
                vmin = min(np.nanmin(real_index), np.nanmin(pred_index))
                vmax = max(np.nanmax(real_index), np.nanmax(pred_index))
                vmin = max(vmin, 0)
                vmax = min(vmax, 1)
            else:
                cmap = 'viridis'
                vmin = min(np.nanmin(real_index), np.nanmin(pred_index))
                vmax = max(np.nanmax(real_index), np.nanmax(pred_index))

            im1 = ax1.imshow(real_index, cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.set_title('Real Hyperspectral', fontsize='xx-large')
            ax1.set_xticks([])
            ax1.set_yticks([])

            im2 = ax2.imshow(pred_index, cmap=cmap, vmin=vmin, vmax=vmax)
            ax2.set_title('Predicted Hyperspectral', fontsize='xx-large')
            ax2.set_xticks([])
            ax2.set_yticks([])
            cbar = plt.colorbar(im2, cax=cax)

            cbar.ax.tick_params(labelsize=16)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Failed to calculate or plot {index_name}: {str(e)}")
