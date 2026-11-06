"""Loss plots grouped by generator architecture.

Contents
--------
plot_generator_comparisons        Validation loss per epoch for MLP, Cramer, CNN, U-Net and ResNet generators.
plot_generator_and_discriminator  Generator / discriminator loss (log10) for the same five architectures.
"""

import numpy as np
import matplotlib.pyplot as plt
from hsi_cgan.visualization.colors import get_color_for_scenario, ordered_scenario_keys, scenario_color_map


# subplots comparing performance across generator architectures
def plot_generator_comparisons(MLP_list, Large_list, CNN_list, UNet_list, ResNet_list):
    fig, axs = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Model Performance by Generator Architecture', fontsize=20)

    axs[2,1].axis('off')

    plot_configs = [
        {'ax': axs[0,0], 'data_list': MLP_list, 'title': 'MLP Generator'},
        {'ax': axs[0,1], 'data_list': Large_list, 'title': 'Cramér Generator'},
        {'ax': axs[1,0], 'data_list': CNN_list, 'title': 'CNN Generator'},
        {'ax': axs[1,1], 'data_list': UNet_list, 'title': 'U-Net Generator'},
        {'ax': axs[2,0], 'data_list': ResNet_list, 'title': 'ResNet Generator'}
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

        for i, (new_model_name, g_loss, d_loss, val_loss, test_loss) in enumerate(data_list):
            epochs = range(1, len(val_loss) + 1)
            color = get_color_for_scenario(new_model_name, scenario_color_map, ordered_scenario_keys)
            ax.plot(epochs, val_loss, label=f'Model {new_model_name}', color=color)
        ax.set_xlabel('Epoch', fontsize='xx-large')
        ax.set_ylabel('Validation Loss', fontsize='xx-large')
        ax.grid(True)
        ax.legend(fontsize='large')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_generator_and_discriminator(MLP_list, Large_list, CNN_list, UNet_list, ResNet_list):
    fig, axs = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Generator and Discriminator Loss by Architecture', fontsize=20)

    axs[2,1].axis('off')

    plot_configs = [
        {'ax': axs[0,0], 'data_list': MLP_list, 'title': 'MLP Generator'},
        {'ax': axs[0,1], 'data_list': Large_list, 'title': 'Cramér Generator'},
        {'ax': axs[1,0], 'data_list': CNN_list, 'title': 'CNN Generator'},
        {'ax': axs[1,1], 'data_list': UNet_list, 'title': 'U-Net Generator'},
        {'ax': axs[2,0], 'data_list': ResNet_list, 'title': 'ResNet Generator'}
    ]

    opacity_level = 0.6

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

        for i, (new_model_name, g_loss, d_loss, val_loss, test_loss) in enumerate(data_list):
            iterations = np.arange(1, len(g_loss) + 1)
            model_color = get_color_for_scenario(new_model_name, scenario_color_map, ordered_scenario_keys)

            new_model_name = new_model_name.split(' ', 1)[1]



            g_loss_np = np.array(g_loss)
            d_loss_np = np.array(d_loss)

            valid_g_indices = g_loss_np > 0
            if np.any(valid_g_indices):
                ax.plot(iterations[valid_g_indices], np.log10(g_loss_np[valid_g_indices]),
                        label=f'{new_model_name} G', color=model_color, linestyle='-', alpha=opacity_level)
            print(len(g_loss_np))

            valid_d_indices = d_loss_np > 0
            if np.any(valid_d_indices):
                ax.plot(iterations[valid_d_indices], np.log10(d_loss_np[valid_d_indices]),
                        label=f'{new_model_name} D', color=model_color, linestyle=':', alpha=opacity_level)

        ax.set_xlabel('Iteration', fontsize='xx-large')
        ax.set_ylabel('Loss ($log_{10}$)', fontsize='xx-large')
        ax.grid(True)
        ax.legend(loc="upper right", fontsize='large')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
