import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from data_loading import load_data, create_dataloader, filter_and_assign_features, calculate_velocities, normalize_velocity, create_sequences
from model_definitions import CNNGenerator1, CNNGenerator2, CNNDiscriminator1, CNNDiscriminator2

# Define constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_channels = 256
BATCH_SIZE = 128
SEQUENCE_LENGTH = 200

# Define data folder path
FOLDER_PATH = '/home/shailendra/Documents/PhD_Oslomet/QuantumGanWaldo/WALDO/WalDo'

# Load and preprocess data
all_files_data = load_data(FOLDER_PATH)
dataset = all_files_data[0]

# Filter and assign features
feature_data = filter_and_assign_features(dataset)
xpl, ypl, xpr, ypr = feature_data['xpl'], feature_data['ypl'], feature_data['xpr'], feature_data['ypr']

# Calculate velocities
velocity_left, velocity_right = calculate_velocities(xpl, ypl, xpr, ypr)

# Prepare velocity data for further processing
velocity_data = pd.DataFrame({
    'velocity_left': velocity_left
})
velocity_data['velocity_left'] = velocity_data['velocity_left'].ffill()  # Handle NaNs

# Normalize and create sequences
velocity_data = normalize_velocity(velocity_data)
subset_velocity_data = velocity_data['normalized_velocity'][:4199]
sequences = create_sequences(subset_velocity_data, SEQUENCE_LENGTH)

# Create DataLoader
train_loader = create_dataloader(sequences, BATCH_SIZE)

# Load the trained models
generator1 = CNNGenerator1(input_channels=256, output_channels=1).to(DEVICE)
generator2 = CNNGenerator2(input_channels=256, output_channels=1).to(DEVICE)

# Ensure the models are in evaluation mode
generator1.eval()
generator2.eval()

# Define periodogram function
def periodogram(tt):
    return torch.log(torch.mean(torch.fft.fft(tt, 200).abs(), 0))

# Function to load scores and return them
def load_scores(model_name):
    js_scores = np.load(f'./results/{model_name}_js_scores.npy')
    mse_scores = np.load(f'./results/{model_name}_mse_scores.npy')
    return js_scores, mse_scores

# Function to visualize results for a model
def visualize_results(generator, train_loader, model_name, input_channels, device):
    # Visualize real and generated periodograms
    real_sample = next(iter(train_loader)).to(device)
    z = torch.randn((real_sample.size(0), input_channels, 1), device=device)
    with torch.no_grad():
        generated_sample = generator(z)

    real_periodogram = periodogram(real_sample)
    generated_periodogram = periodogram(generated_sample)

    # Convert to CPU and detach for plotting
    real_periodogram = real_periodogram.cpu().detach().numpy()
    generated_periodogram = generated_periodogram.cpu().detach().numpy()

    # Ensure the periodogram arrays have enough samples
    num_samples = min(real_periodogram.shape[0], generated_periodogram.shape[0], 4)  # Plot up to 4 samples

    # Plotting
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):  # Plotting up to 4 samples
        plt.subplot(2, 4, i + 1)
        plt.plot(real_periodogram[i].flatten(), label='Real')
        plt.title(f'Real Periodogram - {model_name}')
        plt.subplot(2, 4, i + 5)
        plt.plot(generated_periodogram[i].flatten(), label='Generated')
        plt.title(f'Generated Periodogram - {model_name}')

    plt.tight_layout()
    plt.show()

    # Load the saved scores
    js_scores, mse_scores = load_scores(model_name)

    # Plot and save MSE scores
    plt.figure(figsize=(12, 6))
    plt.plot(mse_scores, label='MSE Score', linewidth=2, color='blue')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('MSE Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./results/{model_name}_mse_scores.pdf')
    plt.show()

    # Plot and save JS scores
    plt.figure(figsize=(12, 6))
    plt.plot(js_scores, label='JS Divergence', linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('JS Divergence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./results/{model_name}_js_scores.pdf')
    plt.show()

    # Fetch a batch of real data from the DataLoader
    real_data = next(iter(train_loader))
    real_data = real_data[:24]  # Select 24 samples for comparison

    # Generate a batch of data using the generator model
    num_samples = 24
    latent_dim = input_channels  # Ensure this matches your latent dimension
    z = torch.randn((num_samples, latent_dim, 1), device=device)
    with torch.no_grad():
        generated_data = generator(z)
    generated_data = generated_data.cpu().detach()[:24]  # Select 24 samples for comparison

    # Flatten the data to determine combined range for min-max binning
    all_data = np.concatenate([real_data.cpu().numpy().flatten(), generated_data.numpy().flatten()])

    # Filter velocities greater than zero
    positive_all_data = all_data[all_data > 0]

    # Compute the log of positive data
    log_all_data = np.log(positive_all_data)

    # Determine bin edges based on the combined range of values
    num_bins = 50
    bin_edges = np.linspace(log_all_data.min(), log_all_data.max(), num_bins + 1)

    # Plotting histograms
    plt.figure(figsize=(15, 10))
    num_plots = 24  # Number of plots to show

    # Plot real and generated data histograms
    for i in range(num_plots):
        plt.subplot(6, 4, i + 1)
        generated_sample = generated_data[i].squeeze().numpy()
        real_sample = real_data[i].squeeze().numpy()

        # Filter out non-positive values
        generated_sample = generated_sample[generated_sample > 0]
        real_sample = real_sample[real_sample > 0]

        # Compute the log of positive data
        log_generated_sample = np.log(generated_sample)
        log_real_sample = np.log(real_sample)
        
        plt.hist(log_real_sample, bins=bin_edges, alpha=0.9, label='Real', color='blue', edgecolor='black')
        plt.hist(log_generated_sample, bins=bin_edges, alpha=0.9, label='Generated', color='red', edgecolor='black')
        plt.title(f'Sequence {i+1}')
        plt.xlabel('Log Scale Velocity')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'./results/{model_name}_histograms.pdf')
    plt.show()

# Call the function for Model 1
visualize_results(generator1, train_loader, 'model_1', input_channels, DEVICE)

# Call the function for Model 2
visualize_results(generator2, train_loader, 'model_2', input_channels, DEVICE)


# Function to generate and compare histograms of real vs generated data
def plot_real_vs_generated_hist(generator, train_loader, latent_dim, model_name, device):
    # Fetch all real data from the DataLoader
    all_real_data = []
    for batch in train_loader:
        all_real_data.append(batch)
    all_real_data = torch.cat(all_real_data)

    # Generate a batch of data using the generator model
    num_samples = len(all_real_data)
    z = torch.randn((num_samples, latent_dim, 1), device=device)
    with torch.no_grad():
        all_generated_data = generator(z)
    all_generated_data = all_generated_data.cpu().detach()

    # Flatten the real and generated data
    flattened_real_data = all_real_data.numpy().flatten()
    flattened_generated_data = all_generated_data.numpy().flatten()

    # Filter velocities greater than zero
    positive_real_data = flattened_real_data[flattened_real_data > 0]
    positive_generated_data = flattened_generated_data[flattened_generated_data > 0]

    # Compute the log of positive data
    log_positive_real_data = np.log(positive_real_data)
    log_positive_generated_data = np.log(positive_generated_data)

    # Determine the range for the bins based on the combined data
    min_value = min(log_positive_real_data.min(), log_positive_generated_data.min())
    max_value = max(log_positive_real_data.max(), log_positive_generated_data.max())

    # Create bins
    num_bins = 40
    bins = np.linspace(min_value, max_value, num_bins + 1)

    # Plotting histograms
    plt.figure(figsize=(10, 6))
    plt.hist(log_positive_real_data, bins=bins, alpha=1, label='Real', color='red', edgecolor='black')
    plt.hist(log_positive_generated_data, bins=bins, alpha=0.7, label='Generated', color='blue', edgecolor='black')
    plt.xlabel('Log Velocity', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14)

    # Fit the plot in the most zoomed form
    plt.xlim(left=-14)
    plt.tight_layout()

    # Save the plot as PDF
    plt.savefig(f'./results/RealVSGenerated_{model_name}_velDCGAN.pdf')
    plt.show()
# Call the function for Model 1
plot_real_vs_generated_hist(generator1, train_loader, latent_dim=256, model_name='model_1', device=DEVICE)

# Call the function for Model 2
plot_real_vs_generated_hist(generator2, train_loader, latent_dim=256, model_name='model_2', device=DEVICE)
