import torch
from torch import nn
from torch.nn import functional as F
from data_loading import load_data, create_sequences, normalize_velocity, filter_and_assign_features, calculate_velocities,create_dataloader 
from model_definitions import CNNGenerator1, CNNGenerator2, CNNDiscriminator1, CNNDiscriminator2, weights_init
from training import train_gan
import pandas as pd
# Define your paths and constants
FOLDER_PATH = '/home/shailendra/Documents/PhD_Oslomet/QuantumGanWaldo/WALDO/WalDo'
SEQUENCE_LENGTH = 200
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
#velocity_data['velocity_left'].ffill(inplace=True)  # Handle NaNs
velocity_data['velocity_left'] = velocity_data['velocity_left'].ffill()  # Handle NaNs
# Normalize the velocity data
velocity_data = normalize_velocity(velocity_data)
# Create sequences from the normalized data
subset_velocity_data = velocity_data['normalized_velocity'][:4199]
sequences = create_sequences(subset_velocity_data, SEQUENCE_LENGTH)
# Prepare DataLoader
train_loader = create_dataloader(sequences, BATCH_SIZE)

# For Model 1
generator1 = CNNGenerator1(input_channels=256, output_channels=1).to(DEVICE)
discriminator1 = CNNDiscriminator1().to(DEVICE)
# For Model 2
generator2 = CNNGenerator2(input_channels=256, output_channels=1).to(DEVICE)
discriminator2 = CNNDiscriminator2().to(DEVICE)
# Initialize weights for both models
generator1.apply(weights_init)
discriminator1.apply(weights_init)
generator2.apply(weights_init)
discriminator2.apply(weights_init)
# Set hyperparameters
lr = 0.0002
input_channels = 256
nb_epoch = 4
# Define periodogram function
def periodogram(tt):
    return torch.log(torch.mean(torch.fft.fft(tt, 200).abs(), 0))
# Train Model 1
train_gan(generator1, discriminator1, train_loader, input_channels, nb_epoch, lr, DEVICE, periodogram, 'model_1')
# Train Model 2
train_gan(generator2, discriminator2, train_loader, input_channels, nb_epoch, lr, DEVICE, periodogram, 'model_2')
