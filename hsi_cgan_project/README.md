# Predicting HSI with conditional GANs: modular version

The original notebook (`predicting_hsi_cgans.ipynb`) has been split into a Python package, `hsi_cgan/`, plus a slim
notebook that keeps only the orchestration code. **All function and class bodies are byte-for-byte the original code**
(verified by comparing the syntax trees). Only imports were added.

```
hsi_cgan_project/
├── README.md
├── predicting_hsi_cgans_modular.ipynb   # orchestration cells (training loop, evaluation loops, plot calls)
└── hsi_cgan/
    ├── models/
    │   ├── generators.py          CNNGenerator, UNetGenerator, ResidualBlock, ResNetGenerator,
    │   │                          GeneratorLarge (+ _MI_PCA, _bandwidth variants), GeneratorMLP
    │   ├── discriminators.py      DiscriminatorMLP, PatchGANDiscriminator, DiscriminatorLarge
    │   └── factory.py             creating_generator_and_discriminator, creating_loss_function, creating_optimizer
    ├── data/
    │   ├── memmap_builder.py      create_concatenated_memmaps
    │   └── dataset.py             HyperspectralDataset (train/val/test split, streaming batches)
    ├── metrics/
    │   ├── losses.py              rmse_loss, mrae_loss
    │   └── image_metrics.py       sam_loss, calculate_metrics, calculate_per_channel_metrics
    ├── training/
    │   ├── checkpoints.py         save_checkpoint, load_checkpoint, save_hyperparameters
    │   ├── validation.py          validate (RMSE/MRAE, used inside the training loop)
    │   └── trainer.py             train
    ├── evaluation/
    │   ├── validators.py          validate, validate_extended, validate_per_channel
    │   ├── checkpoint_loading.py  creating_generator_and_discriminator (eval variant), load_checkpoint (older variant),
    │   │                          extract_model_name, plot_val_losses_from_checkpoints
    │   └── vegetation_indices.py  calculate_vegetation_index
    └── visualization/
        ├── colors.py              scenario colour maps, get_color_for_scenario
        ├── band_plots.py          plot_band_comparisons, plot_specific_band_comparisons,
        │                          plot_specific_band_generator_and_discriminator
        ├── architecture_plots.py  plot_generator_comparisons, plot_generator_and_discriminator
        ├── spectral_plots.py      plot_spectral_curves, visualize_validation_results
        └── vegetation_plots.py    plot_vegetation_index_comparison
```

Every module starts with a docstring listing what it contains.

## Using it
Put the `hsi_cgan/` folder next to the notebook (or add its parent folder to `sys.path`, see the first code cell).
Run the notebook cells in order, as before. Cell 0 still performs the Colab installs and Drive mount.

## Things worth knowing
1. **Duplicate definitions.** The original notebook defined some names more than once; later cells silently overrode
   earlier ones. They are kept as separate versions in separate modules:
   - `creating_generator_and_discriminator`: `models/factory.py` (training) vs `evaluation/checkpoint_loading.py` (also picks the MI/PCA and bandwidth generators)
   - `load_checkpoint`: `training/checkpoints.py` vs `evaluation/checkpoint_loading.py` (older, `val_losses` key)
   - `validate`: `training/validation.py` (RMSE, MRAE) vs `evaluation/validators.py` (adds SAM, PSNR, SSIM)
   - `rmse_loss`, `mrae_loss`, `get_color_for_scenario`, and the colour maps were identical copies and now exist once.

   The notebook imports them in the same order the original cells defined them, so notebook behaviour is unchanged.
   `train()` always uses the training versions.
2. **Notebook globals.** Two unchanged functions read variables that were globals of the notebook:
   `plot_val_losses_from_checkpoints` (`generators_list`, `bands_list`, `dictionary_list`, `hyperparameters`) and
   `plot_spectral_curves` (`hyperparameters`). Python resolves globals per module, so each of those two modules has a
   small `configure(**kwargs)` helper, called once in the notebook (cells 3 and 4).
3. **Top-level scripts stay in the notebook.** The big training/preprocessing loop, the evaluation loop and the
   settings blocks (`dict_names`, `indices_*`, ...) are not functions and differ slightly between cells, so they were
   left exactly as they were.

## Dependencies
torch, torchmetrics, numpy, scipy, matplotlib, scikit-image, tqdm, opencv-python, spectral, optuna, rasterio
(the notebook's setup cell installs the Colab ones).
