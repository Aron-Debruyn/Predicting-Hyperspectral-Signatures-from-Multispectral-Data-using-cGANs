"""Validation / training loss plots grouped by input-band scenario.

Contents
--------
plot_band_comparisons                           Validation loss per epoch for MI, PCA, Centers(+Bandwidth), RGB(+NIR).
plot_specific_band_comparisons                  Validation loss for the 16 two-band combinations (saves a PNG).
plot_specific_band_generator_and_discriminator  Generator / discriminator loss (log10) for the 16 combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
from hsi_cgan.visualization.colors import (get_color_for_scenario, ordered_scenario_keys, scenario_color_map,
                                          new_ordered_scenario_keys, new_scenario_color_map)


#subplots comparing performance across bands
def plot_band_comparisons(MI_list, PCA_list, Bandwidth_list, Micasense_list, fourbands_list, threebands_list):
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Model Performance by Input Band Type', fontsize=16)

    plot_configs = [
        {'ax': axs[0,0], 'data_list': MI_list, 'title': 'MI Input', 'expected_scenario': 'MI'},
        {'ax': axs[0,1], 'data_list': PCA_list, 'title': 'PCA Input', 'expected_scenario': 'PCA'},
        {'ax': axs[1,0], 'data_list': Bandwidth_list, 'title': 'Centers + Bandwidth Input', 'expected_scenario': 'Centers + Bandwidth'},
        {'ax': axs[1,1], 'data_list': Micasense_list, 'title': 'Centers Input', 'expected_scenario': 'Centers'},
        {'ax': axs[2,0], 'data_list': fourbands_list, 'title': 'RGB + NIR Input', 'expected_scenario': 'RGB + NIR'},
        {'ax': axs[2,1], 'data_list': threebands_list, 'title': 'RGB Input', 'expected_scenario': 'RGB'}
    ]

    for config in plot_configs:
        ax = config['ax']
        data_list = config['data_list']
        title = config['title']

        ax.set_title(title)
        if not data_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Loss')
            ax.grid(True)
            continue

        for i, (new_model_name, g_loss, d_loss, val_loss, test_loss) in enumerate(data_list):
            epochs = range(1, len(val_loss) + 1)
            color = get_color_for_scenario(new_model_name, scenario_color_map, ordered_scenario_keys)
            ax.plot(epochs, val_loss, label=f'Model {new_model_name}', color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# subplots comparing performance for specific band combinations.
def plot_specific_band_comparisons(
    B_RE717_list, R_RE717_list, G_RE717_list,
    G_NIR_list, B_NIR_list, R_NIR_list,
    G531_NIR_list, R650_NIR_list, R_RE705_list,
    R_RE740_list, B_RE705_list, B_RE740_list,
    G_RE705_list, G_RE740_list, G531_RE_list, R650_RE_list
):
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle('Model Validation Loss by Input Band Combination', fontsize=20)

    # Combine the lists for each subplot
    r_data = R_RE717_list + R_NIR_list + R_RE705_list + R_RE740_list
    g_data = G_RE717_list + G_NIR_list + G_RE705_list + G_RE740_list
    b_data = B_RE717_list + B_NIR_list + B_RE705_list + B_RE740_list
    misc_data = G531_NIR_list + R650_NIR_list + G531_RE_list + R650_RE_list

    plot_configs = [
        {'ax': axs[0, 0], 'data_list': r_data, 'title': 'Red Channel Combinations'},
        {'ax': axs[0, 1], 'data_list': g_data, 'title': 'Green Channel Combinations'},
        {'ax': axs[1, 0], 'data_list': b_data, 'title': 'Blue Channel Combinations'},
        {'ax': axs[1, 1], 'data_list': misc_data, 'title': 'Specific Wavelength Combinations'}
    ]

    for config in plot_configs:
        ax = config['ax']
        data_list = config['data_list']
        title = config['title']

        ax.set_title(title, fontsize=16)
        if not data_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Loss')
            ax.grid(True)
            continue

        for i, (model_name, g_loss, d_loss, val_loss, test_loss) in enumerate(data_list):
            epochs = range(1, len(val_loss) + 1)
            color = get_color_for_scenario(model_name, new_scenario_color_map, new_ordered_scenario_keys)
            ax.plot(epochs, val_loss, label=f'Model {model_name}', color=color, linewidth=2)

        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Validation Loss', fontsize=14)
        ax.grid(True)
        ax.legend(fontsize='medium')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("specific_band_comparisons.png")


# figure comparing G and D losses for specific band combinations.
def plot_specific_band_generator_and_discriminator(
    B_RE717_list, R_RE717_list, G_RE717_list,
    G_NIR_list, B_NIR_list, R_NIR_list,
    G531_NIR_list, R650_NIR_list, R_RE705_list,
    R_RE740_list, B_RE705_list, B_RE740_list,
    G_RE705_list, G_RE740_list, G531_RE_list, R650_RE_list
):
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle('Generator and Discriminator Loss by Input Band Combination', fontsize=20)


    r_data = R_RE717_list + R_NIR_list + R_RE705_list + R_RE740_list
    g_data = G_RE717_list + G_NIR_list + G_RE705_list + G_RE740_list
    b_data = B_RE717_list + B_NIR_list + B_RE705_list + B_RE740_list
    misc_data = G531_NIR_list + R650_NIR_list + G531_RE_list + R650_RE_list

    plot_configs = [
        {'ax': axs[0, 0], 'data_list': r_data, 'title': 'Red Channel Combinations'},
        {'ax': axs[0, 1], 'data_list': g_data, 'title': 'Green Channel Combinations'},
        {'ax': axs[1, 0], 'data_list': b_data, 'title': 'Blue Channel Combinations'},
        {'ax': axs[1, 1], 'data_list': misc_data, 'title': 'Specific Wavelength Combinations'}
    ]

    opacity_level = 0.7  # Opacity for the plot lines

    for config in plot_configs:
        ax = config['ax']
        data_list = config['data_list']
        title = config['title']

        ax.set_title(title, fontsize=16)
        if not data_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss ($log_{10}$)')
            ax.grid(True)
            continue

        for i, (model_name, g_loss, d_loss, val_loss, test_loss) in enumerate(data_list):
            iterations = np.arange(1, len(g_loss) + 1)

            model_color = get_color_for_scenario(model_name, new_scenario_color_map, new_ordered_scenario_keys)

            g_loss_np = np.array(g_loss)
            d_loss_np = np.array(d_loss)


            valid_g_indices = g_loss_np > 0
            if np.any(valid_g_indices):
                ax.plot(iterations[valid_g_indices], np.log10(g_loss_np[valid_g_indices]),
                        label=f'{model_name} G', color=model_color, linestyle='-', alpha=opacity_level)


            valid_d_indices = d_loss_np > 0
            if np.any(valid_d_indices):
                ax.plot(iterations[valid_d_indices], np.log10(d_loss_np[valid_d_indices]),
                        label=f'{model_name} D', color=model_color, linestyle=':', alpha=opacity_level)

        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Loss ($log_{10}$)', fontsize=14)
        ax.grid(True)
        ax.legend(fontsize='medium')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
