import torch
import numpy as np
import time
import torch.nn as nn
import progressbar
import matplotlib.pyplot as plt
from scipy.special import rel_entr




# Initialize weights as per DCGAN specifications
def weights_init(m):
    """
    This function initializes the model weights randomly from a 
    Normal distribution. Follows the DCGAN paper specification.
    Source: https://arxiv.org/pdf/1511.06434.pdf
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Define periodogram function
def periodogram(tt):
    return torch.log(torch.mean(torch.fft.fft(tt, 200).abs(), 0))


# Function to calculate JS divergence
def js_divergence(P, Q):
    """
    Calculates the Jensen-Shannon divergence between two probability distributions.
    """
    _P = P / np.sum(P)
    _Q = Q / np.sum(Q)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (np.sum(rel_entr(_P, _M)) + np.sum(rel_entr(_Q, _M)))

def train_gan(generator, discriminator, train_loader, input_channels, nb_epoch, lr, device, periodogram, model_name):
    """
    Function to train GAN models (model 1 and model 2) with given parameters.
    """
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss and Optimizers
    criterion = torch.nn.BCELoss()
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    mse_scores, js_scores = [], []
    all_real_data, all_generated_data = [], []

    # Training loop
    time_start = time.perf_counter()
    for epoch in progressbar.progressbar(range(nb_epoch)):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        epoch_mse_scores, epoch_js_scores = [], []

        for batch, x in enumerate(train_loader):
            # Train Discriminator
            x = x.to(device)
            batch_size = x.shape[0]
            target_ones = torch.ones((batch_size, 1), device=device)
            target_zeros = torch.zeros((batch_size, 1), device=device)

            discriminator.zero_grad()
            pred_real = discriminator(x)
            loss_real = criterion(pred_real, target_ones)

            z = torch.randn((batch_size, input_channels, 1), device=device)
            with torch.no_grad():
                fake_samples = generator(z)
            pred_fake = discriminator(fake_samples)
            loss_fake = criterion(pred_fake, target_zeros)

            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optim_d.step()

            loss_d_real_running += loss_real.item()
            loss_d_fake_running += loss_fake.item()

            # Train Generator
            generator.zero_grad()
            z = torch.randn((batch_size, input_channels, 1), device=device)
            generated = generator(z)
            classifications = discriminator(generated)
            loss_g = criterion(classifications, target_ones) + torch.mean((periodogram(generated) - periodogram(x)) ** 2)

            loss_g.backward()
            optim_g.step()

            loss_g_running += loss_g.item()

            # Calculate scores
            mse_score = torch.mean((periodogram(generated) - periodogram(x)) ** 2).item()
            epoch_mse_scores.append(mse_score)

            real_periodogram = periodogram(x).cpu().detach().numpy().flatten()
            generated_periodogram = periodogram(generated).cpu().detach().numpy().flatten()
            js_score = js_divergence(real_periodogram, generated_periodogram)
            epoch_js_scores.append(js_score)

            real_data_np = x.cpu().detach().numpy()
            generated_data_np = generated.cpu().detach().numpy()
            all_real_data.append(real_data_np)
            all_generated_data.append(generated_data_np)

        # Average scores for the epoch
        mse_scores.append(np.mean(epoch_mse_scores))
        js_scores.append(np.mean(epoch_js_scores))

        if epoch % 1 == 0:
            print(f'\nEpoch [{epoch + 1}/{nb_epoch}] G: {np.mean(epoch_mse_scores)}, Dr: {loss_d_real_running / len(train_loader)}, Df: {loss_d_fake_running / len(train_loader)}, JS: {np.mean(epoch_js_scores)}')

            # Generate and plot sample trajectories
            z = torch.randn((8, input_channels, 1), device=device)
            with torch.no_grad():
                generated = generator(z)
            traj_sim = generated.cpu().detach().numpy()

            plt.figure(figsize=(8, 8))
            for i in range(8):
                plt.subplot(4, 4, i + 1)
                data = traj_sim[i, 0, :]
                plt.plot(data, c='black', alpha=0.5)
                plt.scatter(range(len(data)), data, c=np.arange(len(data)), cmap='Blues_r')
                plt.scatter(0, data[0], c='red')
            plt.show()
            plt.close()

    # Save the models and results
    torch.save(generator.state_dict(), f'./results/{model_name}_G.pt')
    torch.save(discriminator.state_dict(), f'./results/{model_name}_D.pt')

    with open(f'./results/{model_name}_mse_scores.npy', 'wb') as f:
        np.save(f, mse_scores)

    with open(f'./results/{model_name}_js_scores.npy', 'wb') as f:
        np.save(f, js_scores)

    np.save(f'./results/{model_name}_real_data.npy', np.concatenate(all_real_data, axis=0))
    np.save(f'./results/{model_name}_generated_data.npy', np.concatenate(all_generated_data, axis=0))
