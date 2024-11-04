# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import DataLoader
from progressbar import ProgressBar
import os

from models import get_generator, get_discriminator, weights_init
from utils import periodogram, js_divergence

def train_gan(train_set, device, save_path, nb_epoch=120, batch_size=128, latent_dim=256,
              generator_model='CNNGenerator', discriminator_model='CNNDiscriminator'):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Instantiate generator
    if generator_model == 'LSTMGenerator':
        generator = get_generator(generator_model, input_channels=latent_dim, output_channels=1).to(device)
    else:
        generator = get_generator(generator_model, input_channels=latent_dim).to(device)

    # Instantiate discriminator
    if discriminator_model == 'LSTMDiscriminator':
        discriminator = get_discriminator(discriminator_model, input_size=1).to(device)
    else:
        discriminator = get_discriminator(discriminator_model).to(device)

    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    lr = 0.0002
    optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training the loop
    mse_scores = []
    js_scores = []

    time_start = time.perf_counter()
    for epoch in ProgressBar()(range(nb_epoch)):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        n_batches = 0

        for batch_idx, x in enumerate(train_loader):
            n_batches += 1
            x = x.to(device)  # x shape: (batch_size, 1, seq_length)
            batch_size, _, seq_length = x.shape

            # Adjust x for discriminator input
            if discriminator_model.startswith('LSTM'):
                x = x.squeeze(1).unsqueeze(2)  # Shape: (batch_size, seq_length, 1)
            elif discriminator_model.startswith('CNN'):
                pass  # x is already in shape (batch_size, 1, seq_length)

            #### TRAIN DISCRIMINATOR #######
            discriminator.zero_grad()

            
            pred_real = discriminator(x)
            target_real = torch.ones((batch_size, 1), device=device)
            loss_real = criterion(pred_real, target_real)

            # Fake data
            if generator_model.startswith('LSTM'):
                z = torch.randn((batch_size, seq_length, latent_dim), device=device)
            elif generator_model.startswith('CNN'):
                z = torch.randn((batch_size, latent_dim, 1), device=device)

            with torch.no_grad():
                generated = generator(z)

            # Adjust generated data for discriminator input
            if discriminator_model.startswith('LSTM'):
                if generator_model.startswith('CNN'):
                    generated = generated.squeeze(1).unsqueeze(2)  # Shape: (batch_size, seq_length, 1)
            elif discriminator_model.startswith('CNN'):
                if generator_model.startswith('LSTM'):
                    generated = generated.permute(0, 2, 1)  # Shape: (batch_size, 1, seq_length)

            pred_fake = discriminator(generated)
            target_fake = torch.zeros((batch_size, 1), device=device)
            loss_fake = criterion(pred_fake, target_fake)

            # Combine losses and update discriminator
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optim_d.step()

            loss_d_real_running += loss_real.item()
            loss_d_fake_running += loss_fake.item()

            #### TRAIN GENERATOR ####
            generator.zero_grad()

            if generator_model.startswith('LSTM'):
                z = torch.randn((batch_size, seq_length, latent_dim), device=device)
            elif generator_model.startswith('CNN'):
                z = torch.randn((batch_size, latent_dim, 1), device=device)

            generated = generator(z)

            # Adjust generated data for discriminator input
            if discriminator_model.startswith('LSTM'):
                if generator_model.startswith('CNN'):
                    generated = generated.squeeze(1).unsqueeze(2)
            elif discriminator_model.startswith('CNN'):
                if generator_model.startswith('LSTM'):
                    generated = generated.permute(0, 2, 1)

            pred_gen = discriminator(generated)
            target_gen = torch.ones((batch_size, 1), device=device)
            loss_g = criterion(pred_gen, target_gen)

            # Periodogram calculation
            if generator_model.startswith('LSTM'):
                periodogram_generated = periodogram(generated.permute(0, 2, 1))  # Shape adjustment if needed
                periodogram_real = periodogram(x.permute(0, 2, 1))
            else:
                periodogram_generated = periodogram(generated)
                periodogram_real = periodogram(x)

            loss_g += 12 * torch.mean((periodogram_generated - periodogram_real) ** 2)

            loss_g.backward()
            optim_g.step()

            loss_g_running += loss_g.item()

            # Calculate scores
            mse_score = torch.mean((periodogram_generated - periodogram_real) ** 2).item()
            mse_scores.append(mse_score)

            # Calculate JS divergence
            real_periodogram = periodogram_real.cpu().detach().numpy().flatten()
            generated_periodogram = periodogram_generated.cpu().detach().numpy().flatten()
            js_score = js_divergence(real_periodogram, generated_periodogram)
            js_scores.append(js_score)

        # Average scores
        avg_mse_score = sum(mse_scores) / len(mse_scores)
        avg_js_score = sum(js_scores) / len(js_scores)

        if epoch % 1 == 0:
            print('\nEpoch [{}/{}] -----------------------------------------------------------------------------'
                  .format(epoch + 1, nb_epoch))
            print('G Loss: {:.4f}, D Real Loss: {:.4f}, D Fake Loss: {:.4f}, JS Divergence: {:.4f}'.format(
                  loss_g_running / n_batches,
                  loss_d_real_running / n_batches,
                  loss_d_fake_running / n_batches,
                  avg_js_score))

    # Save the scores and models
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'mse_scores.npy'), mse_scores)
    np.save(os.path.join(save_path, 'js_scores.npy'), js_scores)
    torch.save(generator.state_dict(), os.path.join(save_path, 'generator.pt'))
    torch.save(discriminator.state_dict(), os.path.join(save_path, 'discriminator.pt'))

    return generator, discriminator

