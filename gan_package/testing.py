# testing.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from models import LSTMGenerator  # Import the LSTMGenerator class

def evaluate_gan(generator, all_real_data, device, save_path, latent_dim=256):
    # Check the shape of all_real_data
    print("Shape of all_real_data:", all_real_data.shape)

    # Determine the sequence length based on the shape
    if len(all_real_data.shape) == 3:
        # Shape: (num_samples, channels, seq_length)
        seq_length = all_real_data.shape[2]
    elif len(all_real_data.shape) == 2:
        seq_length = all_real_data.shape[1]
    else:
        raise ValueError("Unexpected shape for all_real_data: {}".format(all_real_data.shape))

    num_samples = len(all_real_data)
    if isinstance(generator, LSTMGenerator):
        z = torch.randn((num_samples, seq_length, latent_dim), device=device)
    else:
        z = torch.randn((num_samples, latent_dim, 1), device=device)

    with torch.no_grad():
        generated_data = generator(z)
    generated_data = generated_data.cpu().detach()

    # Adjust generated data shape for evaluation
    if isinstance(generator, LSTMGenerator):
        generated_data = generated_data.permute(0, 2, 1)  # Shape: (num_samples, output_channels, seq_length)

    # Flatten the real and generated data
    flattened_real_data = all_real_data.numpy().flatten()
    flattened_generated_data = generated_data.numpy().flatten()

    # Filter velocities greater than zero
    positive_real_data = flattened_real_data[flattened_real_data > 0]
    positive_generated_data = flattened_generated_data[flattened_generated_data > 0]

    # Compute the log of positive data
    log_positive_real_data = np.log(positive_real_data + 1e-8)  # Add epsilon to prevent log(0)
    log_positive_generated_data = np.log(positive_generated_data + 1e-8)

    # Determine the range for the bins
    min_value = min(log_positive_real_data.min(), log_positive_generated_data.min())
    max_value = max(log_positive_real_data.max(), log_positive_generated_data.max())

    num_bins = 60
    bins = np.linspace(min_value, max_value, num_bins + 1)

    # Plotting histogram
    plt.figure(figsize=(10, 6))
    plt.hist(log_positive_real_data, bins=bins, alpha=1, label='Real', color='red', edgecolor='black')
    plt.hist(log_positive_generated_data, bins=bins, alpha=0.7, label='Generated', color='blue', edgecolor='black')
    plt.xlabel('Log Velocity')
    plt.xlim(left=-12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'RealVSGenerated_velGAN.pdf'))
    plt.show()

