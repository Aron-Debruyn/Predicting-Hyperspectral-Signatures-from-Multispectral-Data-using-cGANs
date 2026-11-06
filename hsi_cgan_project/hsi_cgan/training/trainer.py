"""The conditional-GAN training loop.

Contents
--------
train  Trains generator + discriminator (architectures 'rgb' and 'MLP'), validates every epoch, writes
       checkpoints, plots predicted vs. true spectra / error maps, runs the final test pass and saves the
       final model.
"""

from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from hsi_cgan.models.factory import creating_generator_and_discriminator, creating_loss_function, creating_optimizer
from hsi_cgan.training.checkpoints import save_checkpoint, load_checkpoint
from hsi_cgan.training.validation import validate


def train(dataloader, val_dataloader, test_dataloader, hyperparameters):
  generator,discriminator = creating_generator_and_discriminator(name_g=hyperparameters['name_g'],
                                                               name_d=hyperparameters['name_d'],
                                                               input_channels=len(hyperparameters['indices']),
                                                               output_channels=hyperparameters['total_length']-(hyperparameters['exclude_begin']+hyperparameters['exclude_end']),
                                                               hard_code_known=hyperparameters['hard_code_known'])
  def find_closest_band(target_wavelength, wavelength_list):
    return np.abs(wavelength_list - target_wavelength).argmin()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  generator.to(device)
  discriminator.to(device)


  g_optimizer, d_optimizer = creating_optimizer(name_optim=hyperparameters['name_optim'], generator=generator, discriminator=discriminator,
                                                learning_rate=hyperparameters['learning_rate'], betas=hyperparameters['betas'])
  loss = creating_loss_function(name=hyperparameters['name_loss'])
  adversarial_loss = creating_loss_function(name=hyperparameters['name_adversarial_loss'])


  # Decaying shedule !!!!
  # g_scheduler = StepLR(g_optimizer, step_size=hyperparameters['decay_epochs'], gamma=hyperparameters['decay_rate'])
  # d_scheduler = StepLR(d_optimizer, step_size=hyperparameters['decay_epochs'], gamma=hyperparameters['decay_rate'])



  wavelengths= np.load(r"{0}/wavelengthes.npy".format(hyperparameters['basedir']))
  wavelengths = wavelengths[hyperparameters['exclude_begin']:hyperparameters['total_length']-hyperparameters['exclude_end']]
  # Training loop
  g_losses = []
  d_losses = []
  val_losses_rmse = []
  test_losses_rmse = []
  val_losses_mae = []
  test_losses_mae = []
  sample_multispectral, sample_hyperspectral = next(iter(val_dataloader))
  sample_multispectral = sample_multispectral.to(device)

  sample_hyperspectral = sample_hyperspectral.to(device)
  lambda_loss = 100

  if hyperparameters['load_model']:
      start_epoch, g_losses, d_losses, val_losses_rmse, val_losses_mae = load_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                                                                    filename=hyperparameters['checkpoint_path'], hyperparameters=hyperparameters)
      g_optimizer.param_groups[0]['lr'] = hyperparameters['learning_rate']
      d_optimizer.param_groups[0]['lr'] = hyperparameters['learning_rate']
      test_losses = []
  else:
      start_epoch = 0
      g_losses = []
      d_losses = []
      val_losses_rmse = []
      test_losses_rmse = []
      val_losses_mae = []
      test_losses_mae = []

  for epoch in range(start_epoch, hyperparameters['epochs']):
      for i, (ms_images, hs_images) in tqdm(enumerate(dataloader)):
          ms_images, hs_images = ms_images.to(device), hs_images.to(device)
          if hyperparameters['architecture'] == 'rgb':
            d_optimizer.zero_grad()

            fake_hyperspectral_image = generator(ms_images)

            noise_real = torch.randn_like(hs_images) * 0.05
            noise_fake = torch.randn_like(fake_hyperspectral_image) * 0.05
            hs_images_noisy = hs_images + noise_real
            fake_hyperspectral_image_noisy = fake_hyperspectral_image + noise_fake

            real_output = discriminator(ms_images, hs_images_noisy)
            fake_output = discriminator(ms_images, fake_hyperspectral_image_noisy.detach())

            real_loss = adversarial_loss(real_output, torch.ones_like(real_output)*0.9) # one-sided label smoothing
            fake_loss = adversarial_loss(fake_output, torch.zeros_like(fake_output))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()


            g_optimizer.zero_grad()

            fake_output = discriminator(ms_images, fake_hyperspectral_image)
            g_adv_loss = adversarial_loss(fake_output, torch.ones_like(fake_output))
            g_l1_loss = loss(fake_hyperspectral_image, hs_images)
            g_loss = g_adv_loss + lambda_loss * g_l1_loss  # lambda = 100 as in the paper
            g_loss.backward()
            g_optimizer.step()


            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

          elif hyperparameters['architecture'] == 'MLP':

            d_optimizer.zero_grad()


            fake_hyperspectral_image = generator(ms_images)

            noise_real = torch.randn_like(hs_images) * 0.05
            noise_fake = torch.randn_like(fake_hyperspectral_image) * 0.05
            hs_images_noisy = hs_images + noise_real
            fake_hyperspectral_image_noisy = fake_hyperspectral_image + noise_fake

            real_output = discriminator(ms_images, hs_images_noisy)
            fake_output = discriminator(ms_images, fake_hyperspectral_image_noisy.detach())


            real_loss = adversarial_loss(real_output, torch.ones_like(real_output)*0.9)
            fake_loss = adversarial_loss(fake_output, torch.zeros_like(fake_output))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()


            g_optimizer.zero_grad()


            fake_hyperspectral_image = generator(ms_images)


            # pass the concatenated input to the discriminator
            fake_output = discriminator(ms_images, fake_hyperspectral_image)
            g_adv_loss = adversarial_loss(fake_output, torch.ones_like(fake_output))
            g_l1_loss = loss(fake_hyperspectral_image, hs_images)
            g_loss = (g_adv_loss + lambda_loss * g_l1_loss)  # lambda = 100 as in the paper
            g_loss.backward()
            g_optimizer.step()


            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())


      # sheduler !!!!!!!!!!!!!!!!!!!!!
      # g_scheduler.step()
      # d_scheduler.step()

      learning_rate = g_optimizer.param_groups[0]['lr']
      print("New learning rate: ", learning_rate)

      print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
      val_loss_rmse, val_loss_mae = validate(generator, val_dataloader, device, hyperparameters['indices'], hyperparameters['hard_code_known'])
      print(f"Validation Loss RMSE, MAE: {val_loss_rmse}, {val_loss_mae}")
      val_losses_rmse.append(val_loss_rmse)
      val_losses_mae.append(val_loss_mae)

      # saving the generator temporarily
      if epoch % hyperparameters['saving_epoch'] == 0:
        print('saving checkpoint')
        checkpoint_path = f"{hyperparameters['basedir']}/checkpoints/checkpoint_epoch_{epoch+1}_{hyperparameters['name']}_{hyperparameters['Bands']}.pth"
        save_checkpoint(epoch + 1, generator, discriminator, g_optimizer, d_optimizer, g_losses, d_losses,
                        val_losses_rmse, val_losses_mae, checkpoint_path)

      # Visualization: True vs. Predicted after each epoch
      with torch.no_grad():

          if hyperparameters['architecture'] == 'MLP' or hyperparameters['architecture'] == 'cramer':
            predicted_hyperspectral = generator(sample_multispectral).cpu().numpy()[0, :]
            true_hyperspectral_curve = sample_hyperspectral.cpu().numpy()[0, :]

          else:

            if hyperparameters['hard_code_known']:
                predicted_hyperspectral = generator(sample_multispectral, sample_hyperspectral, hyperparameters['indices']).cpu().numpy()[15, :, :, :]
            else:
                predicted_hyperspectral = generator(sample_multispectral)
                error_map, _ = discriminator(sample_multispectral, predicted_hyperspectral, error_map=True)
                true_hyperspectral_image = sample_hyperspectral.cpu().numpy()[15, :, :, :]


                mean_tensor = error_map.mean(dim=(0, 1))
                mean_np = mean_tensor.cpu().numpy()

                mean_np = (mean_np - mean_np.min()) / (mean_np.max() - mean_np.min())


                red_idx = find_closest_band(650, wavelengths)
                green_idx = find_closest_band(550, wavelengths)
                blue_idx = find_closest_band(450, wavelengths)


                print('predicted_hyperspectral shape', true_hyperspectral_image.shape)
                rgb_image = np.stack([
                    true_hyperspectral_image[red_idx, :, :],
                    true_hyperspectral_image[green_idx, :, :],
                    true_hyperspectral_image[blue_idx, :, :]
                ], axis=-1)

                print('rgb_image shape before transpose', rgb_image.shape)

                rgb_image = np.squeeze(rgb_image)

                # Normalize for display
                # rgb_image = rgb_image - np.min(rgb_image)
                # rgb_image /= np.max(rgb_image)


                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())


                plt.figure(figsize=(10, 5))


                plt.subplot(1, 2, 1)
                plt.imshow(rgb_image)
                plt.title("Hyperspectral (RGB)")
                plt.axis('off')

                # Plot error map mean
                plt.subplot(1, 2, 2)
                plt.imshow(mean_np, cmap='viridis')  # Add `vmin=0, vmax=1` if needed
                plt.colorbar()
                plt.title("Error Map Mean (5x5)")
                plt.axis('off')

                plt.tight_layout()
                plt.show()
            predicted_hyperspectral = np.mean(predicted_hyperspectral, axis=(1, 2))

            true_hyperspectral_curve = sample_hyperspectral.cpu().numpy()[15, :, :, :]
            true_hyperspectral_curve = np.mean(true_hyperspectral_curve, axis=(1, 2))

          plt.figure(figsize=(10, 5))
          plt.plot(wavelengths, true_hyperspectral_curve, label="True Hyperspectral", color='green')
          plt.plot(wavelengths, predicted_hyperspectral, label="Predicted Hyperspectral", color='red')
          plt.scatter(wavelengths[hyperparameters['indices']], true_hyperspectral_curve[hyperparameters['indices']],
                      label="Used Bands ({0})".format(len(hyperparameters['indices'])), color='blue') #, marker='*'
          plt.xlabel("Wavelength (nm)")
          plt.ylabel("Reflectance")
          plt.title(f"True vs Predicted Hyperspectral Data (Epoch {epoch+1})")
          plt.legend()
          plt.grid()
          plt.show()

  now = str(datetime.now())
  now = now.replace(" ", "_")


  # Save final checkpoint
  # Final testing after training completes
  test_loss_rmse, test_loss_mae = validate(generator, test_dataloader, device, hyperparameters['indices'], hyperparameters['hard_code_known'])
  test_losses_rmse.append(test_loss_rmse)
  print(f"\nFinal Test Loss: {test_loss_rmse}, {test_loss_mae}")

  # Save final model with test results
  final_checkpoint_path = f"{hyperparameters['basedir']}/models/final_checkpoint_{hyperparameters['name']}_{hyperparameters['Bands']}.pth"
  save_checkpoint(
      hyperparameters['epochs'], generator, discriminator, g_optimizer, d_optimizer,
      g_losses, d_losses, val_losses_rmse+test_losses_rmse, val_losses_mae+test_losses_mae, final_checkpoint_path
  )


  plt.figure(figsize=(10, 5))
  plt.plot(g_losses, label="Generator Loss")
  plt.plot(d_losses, label="Discriminator Loss")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.title("Training Losses")
  plt.show()


  # Visualize Predicted vs True Hyperspectral Data
  with torch.no_grad():


    if hyperparameters['architecture'] == 'MLP' or hyperparameters['architecture'] == 'cramer':
      predicted_hyperspectral = generator(sample_multispectral).cpu().numpy()[0, :]
      true_hyperspectral_curve = sample_hyperspectral.cpu().numpy()[0, :]

    else:
      if hyperparameters['hard_code_known']:
          predicted_hyperspectral = generator(sample_multispectral, sample_hyperspectral, hyperparameters['indices']).cpu().numpy()[15, :, :, :]
      else:
          predicted_hyperspectral = generator(sample_multispectral)
          error_map, _ = discriminator(sample_multispectral, predicted_hyperspectral, error_map=True)
          true_hyperspectral_image = sample_hyperspectral.cpu().numpy()[15, :, :, :]


          mean_tensor = error_map.mean(dim=(0, 1))
          mean_np = mean_tensor.cpu().numpy()

          mean_np = (mean_np - mean_np.min()) / (mean_np.max() - mean_np.min())



          red_idx = find_closest_band(650, wavelengths)
          green_idx = find_closest_band(550, wavelengths)
          blue_idx = find_closest_band(450, wavelengths)

          print('predicted_hyperspectral shape', true_hyperspectral_image.shape)
          rgb_image = np.stack([
              true_hyperspectral_image[red_idx, :, :],
              true_hyperspectral_image[green_idx, :, :],
              true_hyperspectral_image[blue_idx, :, :]
          ], axis=-1)

          print('rgb_image shape before transpose', rgb_image.shape)



          rgb_image = np.squeeze(rgb_image)

          # Normalize for display
          # rgb_image = rgb_image - np.min(rgb_image)
          # rgb_image /= np.max(rgb_image)


          rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())


          plt.figure(figsize=(10, 5))


          plt.subplot(1, 2, 1)
          plt.imshow(rgb_image)
          plt.title("Hyperspectral (RGB)")
          plt.axis('off')


          plt.subplot(1, 2, 2)
          plt.imshow(mean_np, cmap='viridis')  # Add `vmin=0, vmax=1` if needed
          plt.colorbar()
          plt.title("Error Map Mean (5x5)")
          plt.axis('off')

          plt.tight_layout()
          plt.show()
      predicted_hyperspectral = np.mean(predicted_hyperspectral, axis=(1, 2))

      true_hyperspectral_curve = sample_hyperspectral.cpu().numpy()[15, :, :, :]
      true_hyperspectral_curve = np.mean(true_hyperspectral_curve, axis=(1, 2))

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths,true_hyperspectral_curve, label="True Hyperspectral", color='green')
    plt.plot(wavelengths, predicted_hyperspectral, label="Predicted Hyperspectral", color='red')
    plt.scatter(wavelengths[hyperparameters['indices']], true_hyperspectral_curve[hyperparameters['indices']], label="Used Bands", color='blue') #, marker='*'
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("True vs Predicted Hyperspectral Data")
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(val_losses_rmse, label="Validation RMSE")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xticks(range(1, len(val_losses_rmse)+1, 5))
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Validation Losses")
    plt.show()
