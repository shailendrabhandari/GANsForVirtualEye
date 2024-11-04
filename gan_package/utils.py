import torch
import numpy as np
from scipy.special import rel_entr

def periodogram(tt):
    # tt: Tensor of shape (batch_size, features, seq_length)
    if tt.dim() == 3 and tt.size(1) != 1:
        # If shape is (batch_size, features, seq_length), no need to permute
        pass
    elif tt.dim() == 3 and tt.size(1) == 1:
        # If shape is (batch_size, 1, seq_length), no need to permute
        pass
    else:
        tt = tt.unsqueeze(1)  # Ensure there is a features dimension

    periodogram = torch.mean(torch.fft.fft(tt, n=tt.size(-1)).abs(), dim=0)
    return torch.log(periodogram + 1e-8)  # Add epsilon to prevent log(0)

def js_divergence(P, Q):
    _P = P / np.sum(P)
    _Q = Q / np.sum(Q)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (np.sum(rel_entr(_P, _M)) + np.sum(rel_entr(_Q, _M)))

