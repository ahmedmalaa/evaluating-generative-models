from typing import Optional

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader


def rearrange_data(x, x_len, pad_val, eos_val):
    """Take in sequence `x` [dims `(n_samples, max_seq_len, n_features)`, data type `float`] and an array of 
    sequence lengths `x_len` [dims `(n_samples,)`, data type `int`] and return: 
        * a reversed sequence `x_rev`, same dims as `x`, and padded at the same indices as `x`.
        * a reversed and shifted (forward by one) sequence `x_rev_shifted`, same dims as `x`, and padded at the same 
            indices as `x`. Like `x_rev` but sequence elements at x_{t} become x_{t-1}, so element at `t=0` is lost 
            and the element at `t=t_end_of_sequence` is assigned `eos_val`.
    Note that `x` is expected to be padded at the end along the sequence dimension, rather than at the beginning.

    Args:
        x (np.ndarray): sequence data [dims `(n_samples, max_seq_len, n_features)`, data type `float`].
        x_len (np.ndarray): array of sequence lengths [dims `(n_samples,)`, data type `int`].
        pad_val (float): padding value to use in output arrays.
        eos_val (float): end-of-sequence indicator value to use in the output `x_rev_shifted`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x_rev, x_rev_shifted
    """
    x_rev = np.full_like(x, pad_val)
    x_rev_shifted = np.full_like(x, pad_val)
    for idx, l in enumerate(x_len):
        x_rev[idx][:l] = x[idx][:l][::-1].copy()
        x_rev_shifted[idx][:l-1] = x_rev[idx][1:l]
        x_rev_shifted[idx][l-1] = eos_val
    return x_rev, x_rev_shifted


def data_to_tensors(x, x_len, x_rev, x_rev_shifted, float_type, device):
    X = torch.tensor(x, device=device, dtype=float_type)
    X_rev = torch.tensor(x_rev, device=device, dtype=float_type)
    X_rev_shifted = torch.tensor(x_rev_shifted, device=device, dtype=float_type)
    X_len = torch.tensor(x_len, dtype=int)  # CPU by requirement of packing.
    return X, X_len, X_rev, X_rev_shifted


def _generate_dummy_data(n_samples, min_timesteps, max_timesteps, n_features, pad_val, seed):
    np.random.seed(seed)
    
    seq_lengths = np.random.randint(low=min_timesteps, high=max_timesteps+1, size=n_samples)  
    # ^ We assume all features for the same example have same seq length.
    
    data = np.full((n_samples, max_timesteps, n_features), pad_val)
    for i, length in enumerate(seq_lengths):
        generated_data = np.random.randn(length, n_features)
        data[i, 0:length, :] = generated_data
    
    return data, seq_lengths


def generate_dummy_data(
    n_samples: int, 
    min_timesteps: int, 
    max_timesteps: int, 
    n_features: int, 
    pad_val: float, 
    eos_val: float, 
    seed: int, 
    to_tensors: bool,
    float_type: Optional[torch.dtype] = None, 
    device: Optional[torch.device] = None):
    
    x, x_len = _generate_dummy_data(n_samples, min_timesteps, max_timesteps, n_features, pad_val, seed)
    x_rev, x_rev_shifted = rearrange_data(x, x_len, pad_val, eos_val)
    
    if to_tensors:
        x, x_len, x_rev, x_rev_shifted = data_to_tensors(
            x, x_len, x_rev, x_rev_shifted, float_type=float_type, device=device)
    
    return x, x_len, x_rev, x_rev_shifted 


def make_dataloader(data_tensors, **dataloader_kwargs):
    dataset = TensorDataset(*data_tensors)
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataset, dataloader
