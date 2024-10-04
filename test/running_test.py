import torch
import pandas as pd

from data_loading import load_data, create_sequences, normalize_velocity, filter_and_assign_features, calculate_velocities, create_dataloader
from training import train_gan_models, weights_init

# Define your paths and constants
FOLDER_PATH = '/home/shailendra/Documents/PhD_Oslomet/QuantumGanWaldo/WALDO/WalDo'
sequence_length = 20
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
all_files_data = load_data(FOLDER_PATH)
dataset = all_files_data[0]

# Filter and assign features
feature_data = filter_and_assign_features(dataset)
xpl, ypl, xpr, ypr = feature_data['xpl'], feature_data['ypl'], feature_data['xpr'], feature_data['ypr']

# Calculate velocities
velocity_left, velocity_right = calculate_velocities(xpl, ypl, xpr, ypr)

# Prepare the velocity data for further processing
velocity_data = pd.DataFrame({
    'velocity_left': velocity_left
})
velocity_data['velocity_left'] = velocity_data['velocity_left'].ffill()  # Handle NaNs

# Normalize the velocity data
velocity_data = normalize_velocity(velocity_data)

# Create sequences from the normalized data
subset_velocity_data = velocity_data['normalized_velocity'][:4199]
sequences = create_sequences(subset_velocity_data, sequence_length)

# Prepare DataLoader
train_loader = create_dataloader(sequences, BATCH_SIZE)

# Start training all models
train_gan_models(train_loader, DEVICE)
