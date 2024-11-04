import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class VelocityDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        sample = torch.from_numpy(sample).float()  
        sample = sample.unsqueeze(0)  # Add channel dimension
        return sample

def load_and_preprocess_data(folder_path, remove_first_n_points=250, remove_nanbefore=250):
    # Get list of all .txt files in the folder
    files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    # Initialize a list to store the data from all files
    all_data = []

    # Load data from each file
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            data = np.loadtxt(file_path)
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # Parameters
    # Lists to store data
    vS_list = []
    for idx, data in enumerate(all_data):
        if data.shape[0] > data.shape[1]:
            data = data.T
        T_i = data[0]
        X_i = data[1]
        Y_i = data[2]
        saccade_i = data[3]

        # Remove the first 'remove_first_n_points' data points
        T_i = T_i[remove_first_n_points:]
        X_i = X_i[remove_first_n_points:]
        Y_i = Y_i[remove_first_n_points:]
        saccade_i = saccade_i[remove_first_n_points:]

        # Calculate differences
        d_X_i = np.diff(X_i)
        d_Y_i = np.diff(Y_i)
        d_T_i = np.diff(T_i)
        d_S_i = np.sqrt(d_X_i**2 + d_Y_i**2)  # Distance

        # Handle NaNs in differences
        Nan_array = (~np.isnan(d_X_i)) & (~np.isnan(d_Y_i)) & (~np.isnan(d_T_i))
        # Identify indices where NaNs occur
        nan_indices = np.where(~Nan_array)[0]
        nan_mask = np.zeros_like(Nan_array, dtype=bool)
        for nan_idx in nan_indices:
            start = max(0, nan_idx - remove_nanbefore)
            end = min(len(Nan_array), nan_idx + remove_nanbefore + 1)
            nan_mask[start:end] = True

        valid_indices = ~nan_mask
        dX_i_filtered = d_X_i[valid_indices]
        dY_i_filtered = d_Y_i[valid_indices]
        dT_i_filtered = d_T_i[valid_indices]
        dS_i_filtered = d_S_i[valid_indices]

        dT_i_filtered_sec = dT_i_filtered / 1000

        # Remove zero or negative time intervals
        valid_time = dT_i_filtered_sec > 0
        dX_i_filtered = dX_i_filtered[valid_time]
        dY_i_filtered = dY_i_filtered[valid_time]
        dT_i_filtered_sec = dT_i_filtered_sec[valid_time]
        dS_i_filtered = dS_i_filtered[valid_time]

        # Calculate velocities
        vS_i = dS_i_filtered / dT_i_filtered_sec

        # Remove non-positive velocities
        valid_velocity = vS_i > 0
        vS_i = vS_i[valid_velocity]

        # Append velocities to the list
        vS_list.append(vS_i)

    # Combine all velocities into one array
    vS_all = np.concatenate(vS_list)

    return vS_all

def prepare_datasets(vS_all, sequence_length=200, num_sequences=11000, batch_size=128):
    # Ensure vS_all is available and handle any NaNs
    if len(vS_all) == 0:
        raise ValueError("No velocities available.")
    else:
        # Convert to DataFrame for preprocessing
        velocity_data = pd.DataFrame({'velocity': vS_all})

        # Fill NaNs forward
        velocity_data['velocity'] = velocity_data['velocity'].ffill()

        # Normalize the data
        scaler = MinMaxScaler()
        velocity_array = velocity_data['velocity'].values.reshape(-1, 1)
        velocity_data_normalized = scaler.fit_transform(velocity_array)
        velocity_data['normalized_velocity'] = velocity_data_normalized.flatten()

        # Sample sequences
        sequences = sample_random_sequences(velocity_data['normalized_velocity'].values,
                                            sequence_length, num_sequences)

        # Split into training and test sets
        train_sequences, test_sequences = train_test_split(
            sequences, test_size=0.2, shuffle=True, random_state=42
        )

        # Instantiate datasets
        train_set = VelocityDataset(train_sequences)
        test_set = VelocityDataset(test_sequences)

        return train_set, test_set, scaler

def sample_random_sequences(data, seq_length, num_sequences):
    sequences = []
    for _ in range(num_sequences):
        start_idx = np.random.randint(0, len(data) - seq_length)
        seq = data[start_idx:start_idx + seq_length]
        sequences.append(seq)
    return np.array(sequences)