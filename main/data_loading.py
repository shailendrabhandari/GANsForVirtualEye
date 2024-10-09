import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Data Loading
def load_data(folder_path):
    """
    Loads all .txt files from the specified folder and returns them as a list of NumPy arrays.
    """
    files = os.listdir(folder_path)
    all_files_data = []

    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            data = np.loadtxt(file_path)
            all_files_data.append(data)

    return all_files_data

# Data Preprocessing
def filter_and_assign_features(dataset, index_to_remove=7):
    """
    Filters the dataset by removing the row at the specified index and assigns feature names.
    """
    filtered_dataset = np.delete(dataset, index_to_remove, axis=0)
    feature_names = ['time', 'xpl', 'ypl', 'psl', 'xpr', 'ypr', 'psr', 'xr', 'yr']
    feature_data = {feature: filtered_dataset[i, :] for i, feature in enumerate(feature_names)}
    return feature_data

def calculate_velocities(xpl, ypl, xpr, ypr, dt=1/1000):
    """
    Calculates the velocities for the left and right eye based on their X and Y positions.
    """
    delta_xpl = np.diff(xpl, prepend=xpl[0])
    delta_ypl = np.diff(ypl, prepend=ypl[0])
    delta_xpr = np.diff(xpr, prepend=xpr[0])
    delta_ypr = np.diff(ypr, prepend=ypr[0])

    distances_left = np.sqrt(delta_xpl**2 + delta_ypl**2)
    distances_right = np.sqrt(delta_xpr**2 + delta_ypr**2)

    velocity_left = distances_left / dt
    velocity_right = distances_right / dt

    return velocity_left, velocity_right

def remove_blink_saccades(velocity_data, blink_removal_window=50):
    """
    Removes data points around blinks by expanding 50 ms before and after each blink.
    """
    # Forward fill missing values to handle NaNs
    velocity_data['velocity_left'] = velocity_data['velocity_left'].ffill()

    # Identify blink positions where NaN values were originally present in velocity_left
    blink_indices = velocity_data.index[velocity_data['velocity_left'].isna()].tolist()

    # Create a set of indices to remove, expanding each blink index by the specified window
    indices_to_remove = set()
    for blink_index in blink_indices:
        start_index = max(0, blink_index - blink_removal_window)
        end_index = min(len(velocity_data) - 1, blink_index + blink_removal_window)
        indices_to_remove.update(range(start_index, end_index + 1))

    # Filter out the identified indices from the dataset
    velocity_data_filtered = velocity_data.drop(indices_to_remove).reset_index(drop=True)
    return velocity_data_filtered

def normalize_velocity(velocity_data):
    """
    Normalizes the velocity data using MinMaxScaler and returns a normalized DataFrame.
    """
    scaler = MinMaxScaler()
    velocity_array = velocity_data.values.reshape(-1, 1)
    velocity_normalized = scaler.fit_transform(velocity_array)
    velocity_data['normalized_velocity'] = velocity_normalized.flatten()

    return velocity_data

def create_sequences(data, seq_length):
    """
    Creates sequences of the specified length from the data.
    """
    if isinstance(data, pd.Series):
        data = data.values
    num_sequences = len(data) - seq_length + 1
    if num_sequences > 0:
        return np.lib.stride_tricks.as_strided(
            data,
            shape=(num_sequences, seq_length),
            strides=(data.strides[0], data.strides[0])
        )
    else:
        return np.array([])

class VelocityDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        sample = torch.from_numpy(sample).float()  # Convert to Float
        sample = sample.unsqueeze(0)  # Add channel dimension
        return sample

def create_dataloader(sequences, batch_size=128, num_workers=0, prefetch_factor=None):
    """
    Creates a DataLoader for the velocity dataset. 
    Set prefetch_factor to None when num_workers is 0.
    """
    train_set = VelocityDataset(sequences)

    # Set prefetch_factor only if num_workers > 0
    if num_workers == 0:
        prefetch_factor = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    return train_loader

# Ensure all functions are exported
__all__ = [
    'load_data',
    'create_sequences',
    'normalize_velocity',
    'filter_and_assign_features',
    'calculate_velocities',
    'remove_blink_saccades',
    'create_dataloader'
]
