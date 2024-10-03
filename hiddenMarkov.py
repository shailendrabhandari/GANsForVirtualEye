import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(velocity_left, num_points=8000):
    """
    Preprocesses the velocity_left data by removing NaNs and normalizing.
    
    Args:
        velocity_left (array-like): The velocity data.
        num_points (int): Number of data points to keep.

    Returns:
        pd.DataFrame: Cleaned and normalized data.
    """
    data = pd.DataFrame({'velocity_left': velocity_left})
    cleaned_data = data.dropna(subset=['velocity_left']).copy()

    if len(cleaned_data) >= num_points:
        cleaned_data = cleaned_data.iloc[:num_points]
    else:
        raise ValueError(f"Not enough data points after removing NaNs. Required: {num_points}")

    # Normalize the 'velocity_left' data
    scaler = MinMaxScaler()
    cleaned_data['velocity_left'] = scaler.fit_transform(cleaned_data[['velocity_left']])

    # Verify no NaN values remain
    if cleaned_data['velocity_left'].isna().any():
        raise ValueError("NaN values detected, check data input or filling strategy.")

    return cleaned_data

def train_hmm_2_hidden_states(velocity_left):
    """
    Trains a Hidden Markov Model (HMM) with 2 hidden states on the velocity data.

    Args:
        velocity_left (array-like): The velocity data.

    Returns:
        tuple: Trained HMM, generated data, and states.
    """
    cleaned_data = preprocess_data(velocity_left)
    
    # Define the HMM with 2 hidden states
    HMmodel_2HS = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100000, init_params='e')
    HMmodel_2HS.startprob_ = np.array([0.8, 0.2])
    HMmodel_2HS.transmat_ = np.array([
        [0.8, 0.2],
        [0.3, 0.7]
    ])

    # Prepare data for fitting
    observations = cleaned_data['velocity_left'].values.reshape(-1, 1)

    # Fit the HMM to the velocity data
    HMmodel_2HS.fit(observations)

    # Generate data and states from the model
    HMgenerated_data_2HS, states = HMmodel_2HS.sample(len(cleaned_data))
    HMgenerated_data_2HS = HMgenerated_data_2HS[HMgenerated_data_2HS > 0]
    log_HMgenerated_data_2HS = np.log(HMgenerated_data_2HS)

    return HMmodel_2HS, HMgenerated_data_2HS, log_HMgenerated_data_2HS, states

def train_hmm_3_hidden_states(velocity_left):
    """
    Trains a Hidden Markov Model (HMM) with 3 hidden states on the velocity data.

    Args:
        velocity_left (array-like): The velocity data.

    Returns:
        tuple: Trained HMM, generated data, and states.
    """
    cleaned_data = preprocess_data(velocity_left)

    # Define the HMM with 3 hidden states
    HMmodel_3HS = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100000, init_params='e')
    HMmodel_3HS.startprob_ = np.array([0.7, 0.2, 0.1])
    HMmodel_3HS.transmat_ = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])

    # Prepare data for fitting
    observations = cleaned_data['velocity_left'].values.reshape(-1, 1)

    # Fit the HMM to the velocity data
    HMmodel_3HS.fit(observations)

    # Generate data and states from the model
    HMgenerated_data_3HS, states = HMmodel_3HS.sample(len(cleaned_data))
    HMgenerated_data_3HS = HMgenerated_data_3HS[HMgenerated_data_3HS > 0]
    log_HMgenerated_data_3HS = np.log(HMgenerated_data_3HS)

    return HMmodel_3HS, HMgenerated_data_3HS, log_HMgenerated_data_3HS, states
