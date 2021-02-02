from typing import Optional

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from .seq2seq_autoencoder import init_hidden


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


def inference_data_to_tensors(x, x_len, float_type, device):
    X = torch.tensor(x, device=device, dtype=float_type)
    X_len = torch.tensor(x_len, dtype=int)  # CPU by requirement of packing.
    return X, X_len


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


def _hc_repr_to_np(hc_repr):
    h, c = hc_repr
    batch_size = h.shape[1]
    h, c = h.view(batch_size, -1), c.view(batch_size, -1)
    h, c = h.detach().cpu().numpy(), c.detach().cpu().numpy()
    hc = np.hstack([h, c])
    return hc


def get_embeddings(seq2seq, dataloaders, padding_value, max_seq_len):
    """Put together the embeddings: stack horizontally the arrays of h and c; stack vertically these arrays.
    """
    hc_np_list = []
    for dataloader in dataloaders:
        seq2seq.eval()
        with torch.no_grad():
            for iter_, dataloader_items in enumerate(dataloader):
                x, x_len = dataloader_items[0], dataloader_items[1]
                batch_size = x.shape[0]
                hc_init = init_hidden(
                    batch_size=batch_size, 
                    hidden_size=seq2seq.encoder.hidden_size, 
                    num_rnn_layers=seq2seq.encoder.num_rnn_layers, 
                    device=x.device)
                hc_repr = seq2seq.get_embeddings_only(
                    x_enc=x, 
                    x_seq_lengths=x_len, 
                    hc_init=hc_init, 
                    padding_value=padding_value, 
                    max_seq_len=max_seq_len)
                hc_np = _hc_repr_to_np(hc_repr)
                hc_np_list.append(hc_np)
    hc_all = np.vstack(hc_np_list)
    return hc_all
