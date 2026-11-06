"""Spectral-curve and per-band metric plots.

Contents
--------
plot_spectral_curves          Mean true vs. predicted spectrum with bootstrap confidence bands.
visualize_validation_results  2x2 bar charts of RMSE / MRAE / PSNR / SSIM per spectral band.

Notebook globals
----------------
plot_spectral_curves (unchanged from the notebook) reads the global name ``hyperparameters`` (for ``basedir``).
Hand it over once before calling validate_extended():

    from hsi_cgan.visualization import spectral_plots
    spectral_plots.configure(hyperparameters=hyperparameters)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

hyperparameters = None


def configure(**kwargs):
    """Inject notebook-level globals that the (unchanged) functions below read."""
    globals().update(kwargs)


def plot_spectral_curves(true_mean, pred_mean, true_ci_low, true_ci_high, pred_ci_low, pred_ci_high, indices):
    wavelengths = np.load(r"{0}/wavelengthes.npy".format(hyperparameters['basedir']))
    wavelengths = wavelengths[35:272-40]




    plt.figure(figsize=(12, 6))
    plt.scatter(wavelengths[indices], true_mean[indices], label="Used Bands", color='blue', s=20)

    plt.plot(wavelengths, pred_mean, label="Predicted Mean", color='red')
    plt.fill_between(
        wavelengths,
        pred_ci_low,
        pred_ci_high,
        alpha=0.2,
        color='red',
        label="Predicted 95% CI (Bootstrap)"
    )



    plt.plot(wavelengths, true_mean, label="True Mean", color='green')
    plt.fill_between(
        wavelengths,
        true_ci_low,
        true_ci_high,
        alpha=0.2,
        color='green',
        label="True 95% CI (Bootstrap)"
    )


    plt.xlabel("Spectral Band", fontsize="xx-large")
    plt.ylabel("Reflectance", fontsize="xx-large")
    plt.title("Mean Spectral Curves with Bootstrapped Confidence Intervals", fontsize="xx-large")
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()


def visualize_validation_results(results, channel_names, test_loader_len, indices):
    tick_places_indices = [0, 38, 86, 108, 164]

    metrics = ['RMSE', 'MRAE', 'PSNR', 'SSIM']
    units = {
        'RMSE': 'Error',
        'MRAE': 'Relative Error',
        'PSNR': 'dB',
        'SSIM': 'Score'
    }

    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for ax, metric in zip(axes, metrics):
        if f'{metric}_per_channel' not in results:
            continue

        y = results[f'{metric}_per_channel']
        x = range(len(y)) if channel_names is None else channel_names

        if metric == 'SSIM':
            valid_mask = ~np.isnan(y)
            if not any(valid_mask):
                ax.text(0.5, 0.5, 'SSIM Not Available', ha='center', va='center')
                ax.set_title('SSIM per Band', fontsize='x-large')
                continue

            y = np.array(y)[valid_mask]
            if channel_names is not None:
                x = np.array(channel_names)[valid_mask]
            else:
                x = np.arange(len(y))


            bar_width = np.min(np.diff(x)) * 0.8 if len(x) > 1 else 1.0
            bars = ax.bar(x, y, alpha=0.7, width=bar_width)


            for i, bar in enumerate(bars):
                if i in indices:
                    bar.set_color('orange')
                else:
                    bar.set_color((0.122, 0.467, 0.706, 0.9))

            mean_val = np.mean(y)
            ax.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f} (N=700)')
            ax.legend(fontsize='large', loc='lower left')
        else:
            bar_width = np.min(np.diff(x)) * 0.8 if channel_names is not None and len(x) > 1 else 0.8
            bars = ax.bar(x, y, alpha=0.7, width=bar_width)

            for i, bar in enumerate(bars):
                if i in indices:
                    bar.set_color('orange')
                else:
                    bar.set_color((0.122, 0.467, 0.706, 0.9))

            mean_val = np.mean(y)
            ax.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f} (N=700)')
            ax.legend(fontsize='large')

        ax.set_title(f'{metric} per Band', fontsize='x-large')
        ax.set_xlabel('Band', fontsize='x-large')
        ax.set_ylabel(units[metric], fontsize='x-large')
        ax.grid(axis='y', linestyle='--')

        if channel_names is not None:

            desired_tick_values = [channel_names[i] for i in tick_places_indices]


            if metric == 'SSIM':
                plotted_values = x
                final_tick_values = [val for val in desired_tick_values if val in plotted_values]
            else:
                final_tick_values = desired_tick_values

            final_tick_labels = [f"{val:.0f}" for val in final_tick_values]

            ax.set_xticks(final_tick_values)
            ax.set_xticklabels(final_tick_labels,  ha="center", fontsize=10)


    plt.tight_layout(pad=3.0)
    plt.show()
