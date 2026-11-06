"""Re-creating trained networks from saved ``.pth`` checkpoints.

Contents
--------
creating_generator_and_discriminator  Evaluation variant: also picks GeneratorLarge_MI_PCA (20 inputs) and
                                      GeneratorLarge_bandwidth (56 inputs).
load_checkpoint                       Older checkpoint reader (expects a single ``val_losses`` key).
extract_model_name                    Derive the model name from a checkpoint filename.
plot_val_losses_from_checkpoints      Load every checkpoint in a folder, rebuild the generators, and group the
                                      loss histories by band scenario and generator architecture. Returns
                                      networks_created_list plus the per-scenario / per-architecture lists.

Notebook globals
----------------
plot_val_losses_from_checkpoints (unchanged from the notebook) reads the global names ``generators_list``,
``bands_list``, ``dictionary_list`` and ``hyperparameters``. Hand them over once before calling it:

    from hsi_cgan.evaluation import checkpoint_loading
    checkpoint_loading.configure(generators_list=generators_list, bands_list=bands_list,
                                 dictionary_list=dictionary_list, hyperparameters=hyperparameters)
"""

import os
import torch
from hsi_cgan.models.generators import (CNNGenerator, UNetGenerator, ResNetGenerator, GeneratorLarge,
                                       GeneratorLarge_MI_PCA, GeneratorLarge_bandwidth, GeneratorMLP)
from hsi_cgan.models.discriminators import PatchGANDiscriminator, DiscriminatorLarge, DiscriminatorMLP

generators_list = None
bands_list = None
dictionary_list = None
hyperparameters = None


def configure(**kwargs):
    """Inject notebook-level globals that the (unchanged) functions below read."""
    globals().update(kwargs)


def creating_generator_and_discriminator(name_g, name_d, input_channels, output_channels, hard_code_known):
  if name_g == 'CNNGenerator':
    generator = CNNGenerator(input_channels, output_channels)
  elif name_g == 'UNetGenerator':
    generator = UNetGenerator(input_channels, output_channels)
  elif name_g == 'ResNetGenerator':
    generator = ResNetGenerator(input_channels, output_channels)
  elif name_g == 'GeneratorLarge' and input_channels <= 5:
    generator = GeneratorLarge(input_channels, output_channels)
  elif name_g == 'GeneratorLarge' and input_channels == 20:
    generator = GeneratorLarge_MI_PCA(input_channels, output_channels)

  elif name_g == 'GeneratorLarge' and input_channels == 56:
    generator = GeneratorLarge_bandwidth(input_channels, output_channels)
  elif name_g == 'MLP':
    generator = GeneratorMLP(input_channels, output_channels)


  if name_d == 'PatchGANDiscriminator':
    discriminator = PatchGANDiscriminator(input_channels, output_channels)
  elif name_d == 'DiscriminatorLarge':
    discriminator = DiscriminatorLarge(input_channels, output_channels)
  elif name_d == 'MLP':
    discriminator = DiscriminatorMLP(input_channels, output_channels)

  return generator, discriminator


def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, filename, hyperparameters):
    checkpoint = torch.load("{0}/checkpoints/".format(hyperparameters['basedir'])+filename,  map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    g_losses = checkpoint['g_losses']
    d_losses = checkpoint['d_losses']
    val_losses = checkpoint['val_losses']
    print(f"Checkpoint loaded from {filename}")
    return epoch, g_losses, d_losses, val_losses


def extract_model_name(filename):
    if filename.startswith('final_checkpoint_'):
        model_name = filename.replace('final_checkpoint_', '').replace('.pth', '')
    elif filename.startswith('checkpoint_epoch_'):
        parts = filename.split('_')
        model_name = '_'.join(parts[4:]).replace('.pth', '')
    else:
        model_name = filename.replace('.pth', '')
    return model_name


def plot_val_losses_from_checkpoints(checkpoint_dir):
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

    if not checkpoint_files:
        print(f"No .pth files found in {checkpoint_dir}")
        return

    # Initialize figure
    networks_created_list = []

    MI_list = [] #tuple (gen_loss, dis_loss, val_loss, test_loss)
    PCA_list = []
    Bandwidth_list = []
    Micasense_list = []
    fourbands_list = []
    threebands_list = []

    B_RE717_list = []
    R_RE717_list = []
    G_RE717_list = []
    G_NIR_list = []
    B_NIR_list = []
    R_NIR_list = []
    G531_NIR_list = []
    R650_NIR_list = []
    R_RE705_list = []
    R_RE740_list = []
    B_RE705_list = []
    B_RE740_list = []
    G_RE705_list = []
    G_RE740_list = []
    G531_RE_list = []
    R650_RE_list = []

    MLP_list = []
    Large_list = []
    CNN_list = []
    UNet_list = []
    ResNet = []

    # Process each checkpoint file
    count = 0
    for checkpoint_file in checkpoint_files:
        # try:
            # Load checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            # Get validation losses
            # Key you want to check
            key_to_check = 'val_losses_rmse'  # Example key (common in PyTorch checkpoints)

            # Check if the key exists
            if key_to_check in checkpoint:
              val_losses = checkpoint['val_losses_rmse']
              g_losses = checkpoint['g_losses']
              d_losses = checkpoint['d_losses']
            else:
              val_losses = checkpoint['val_losses']
              g_losses = checkpoint['g_losses']
              d_losses = checkpoint['d_losses']



            test_loss = val_losses.pop()


            # Get clean model name
            model_name = extract_model_name(checkpoint_file)

            print(f"Test loss of {model_name}: {test_loss}")

            for generator_name in generators_list:
              if generator_name in model_name:
                for band in bands_list:
                  if band in model_name:
                    # creating the generator and discriminator
                    generator,discriminator = creating_generator_and_discriminator(name_g=generator_name,
                                                                name_d=dictionary_list[generator_name][2],
                                                                input_channels=dictionary_list[band],
                                                                output_channels=hyperparameters['total_length']-(hyperparameters['exclude_begin']+hyperparameters['exclude_end']),
                                                                hard_code_known=hyperparameters['hard_code_known'])
                    generator.load_state_dict(checkpoint['generator_state_dict'])
                    # discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

                    # ['CNNGenerator', 'ResNetGenerator','UNetGenerator', 'GeneratorLarge', 'MLP']

                    if generator_name == "GeneratorLarge":
                      generator_name = "Cramér"
                    elif generator_name == "MLP":
                      generator_name = "MLP"
                    elif generator_name == "CNNGenerator":
                      generator_name = "CNN"
                    elif generator_name == "UNetGenerator":
                      generator_name = "U-Net"
                    elif generator_name == "ResNetGenerator":
                      generator_name = "ResNet"

                    # ['MI', 'PCA', 'bandwidth', 'micasense', '4bands', '3bands']

                    if band == "MI":
                      band = "MI"
                    elif band == "PCA":
                      band = "PCA"
                    elif band == "bandwidth":
                      band = "Centers + Bandwidth"
                    elif band == "micasense":
                      band = "Centers"
                    elif band == "4bands":
                      band = "RGB + NIR"
                    elif band == "3bands":
                      band = "RGB"
                    elif band == "B-RE717":
                      band = "B-RE717"
                    elif band == "R-RE717":
                      band = "R-RE717"
                    elif band == "G-RE717":
                      band = "G-RE717"
                    elif band == "G-NIR":
                      band = "G-NIR"
                    elif band == "B-NIR":
                      band = "B-NIR"
                    elif band == "R-NIR":
                      band = "R-NIR"
                    elif band == "G531-NIR":
                      band = "G531-NIR"
                    elif band == "R650-NIR":
                      band = "R650-NIR"
                    elif band == "R-RE705":
                      band = "R-RE705"
                    elif band == "R-RE740":
                      band = "R-RE740"
                    elif band == "B-RE705":
                      band = "B-RE705"
                    elif band == "B-RE740":
                      band = "B-RE740"
                    elif band == "G-RE705":
                      band = "G-RE705"
                    elif band == "G-RE740":
                      band = "G-RE740"
                    elif band == "G531-RE":
                      band = "G531-RE"
                    elif band == "R650-RE":
                      band = "R650-RE"



                    new_model_name = generator_name +" " + band
                    print(f"gen.and dis. {count} are {model_name}")
                    print(f"new model name is {new_model_name}")
                    networks_created_list.append((new_model_name, generator, discriminator))
                    # creating optimizers
                    # g_optimizer, d_optimizer = creating_optimizer(name_optim=hyperparameters['name_optim'], generator=generator, discriminator=discriminator,
                    #                             learning_rate=hyperparameters['learning_rate'], betas=hyperparameters['betas'])

                    # creating loss functions
                    # loss = creating_loss_function(name=hyperparameters['name_loss'])
                    # adversarial_loss = creating_loss_function(name=hyperparameters['name_adversarial_loss'])


                    if band == "MI":
                      print(f"In {band} list, element : ", len(MI_list), f" is {model_name}")
                      MI_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "PCA":
                      print(f"In {band} list, element : ", len(PCA_list), f" is {model_name}")
                      PCA_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "Centers + Bandwidth":
                      print(f"In {band} list, element : ", len(Bandwidth_list), f" is {model_name}")
                      Bandwidth_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "Centers":
                      print(f"In {band} list, element : ", len(Micasense_list), f" is {model_name}")
                      Micasense_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "RGB + NIR":
                      print(f"In {band} list, element : ", len(fourbands_list), f" is {model_name}")
                      fourbands_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "RGB":
                      print(f"In {band} list, element : ", len(threebands_list), f" is {model_name}")
                      threebands_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))


                    elif band == "B-RE717":
                      print(f"In {band} list, element : ", len(B_RE717_list), f" is {model_name}")
                      B_RE717_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "R-RE717":
                      print(f"In {band} list, element : ", len(R_RE717_list), f" is {model_name}")
                      R_RE717_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "G-RE717":
                      print(f"In {band} list, element : ", len(G_RE717_list), f" is {model_name}")
                      G_RE717_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "G-NIR":
                      print(f"In {band} list, element : ", len(G_NIR_list), f" is {model_name}")
                      G_NIR_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "B-NIR":
                      print(f"In {band} list, element : ", len(B_NIR_list), f" is {model_name}")
                      B_NIR_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "R-NIR":
                      print(f"In {band} list, element : ", len(R_NIR_list), f" is {model_name}")
                      R_NIR_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "G531-NIR":
                      print(f"In {band} list, element : ", len(G531_NIR_list), f" is {model_name}")
                      G531_NIR_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "R650-NIR":
                      print(f"In {band} list, element : ", len(R650_NIR_list), f" is {model_name}")
                      R650_NIR_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "R-RE705":
                      print(f"In {band} list, element : ", len(R_RE705_list), f" is {model_name}")
                      R_RE705_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "R-RE740":
                      print(f"In {band} list, element : ", len(R_RE740_list), f" is {model_name}")
                      R_RE740_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "B-RE705":
                      print(f"In {band} list, element : ", len(B_RE705_list), f" is {model_name}")
                      B_RE705_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "B-RE740":
                      print(f"In {band} list, element : ", len(B_RE740_list), f" is {model_name}")
                      B_RE740_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "G-RE705":
                      print(f"In {band} list, element : ", len(G_RE705_list), f" is {model_name}")
                      G_RE705_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "G-RE740":
                      print(f"In {band} list, element : ", len(G_RE740_list), f" is {model_name}")
                      G_RE740_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "G531-RE":
                      print(f"In {band} list, element : ", len(G531_RE_list), f" is {model_name}")
                      G531_RE_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif band == "R650-RE":
                      print(f"In {band} list, element : ", len(R650_RE_list), f" is {model_name}")
                      R650_RE_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))



                    if generator_name == "MLP":
                      print(f"In {generator_name} list, element : ", len(MLP_list), f" is {model_name}")
                      MLP_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif generator_name == "Cramér":
                      print(f"In {generator_name} list, element : ", len(Large_list), f" is {model_name}")
                      Large_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif generator_name == "CNN":
                      print(f"In {generator_name} list, element : ", len(CNN_list), f" is {model_name}")
                      CNN_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif generator_name == "U-Net":
                      print(f"In {generator_name} list, element : ", len(UNet_list), f" is {model_name}")
                      UNet_list.append((new_model_name, g_losses, d_losses, val_losses, test_loss))
                    elif generator_name == "ResNet":
                      print(f"In {generator_name} list, element : ", len(ResNet), f" is {model_name}")
                      ResNet.append((new_model_name, g_losses, d_losses, val_losses, test_loss))




                    count += 1

            print("\n")
    print('done with loading')
    return networks_created_list,MI_list, PCA_list, Bandwidth_list, Micasense_list, fourbands_list, threebands_list, MLP_list, Large_list, CNN_list, UNet_list, ResNet, B_RE717_list, R_RE717_list, G_RE717_list, G_NIR_list, B_NIR_list, R_NIR_list, G531_NIR_list, R650_NIR_list, R_RE705_list, R_RE740_list, B_RE705_list, B_RE740_list, G_RE705_list, G_RE740_list, G531_RE_list, R650_RE_list
