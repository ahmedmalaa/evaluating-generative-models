"""Seq-2-Seq autoencoder.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers, batch_first=True)

    def forward(self, x, x_seq_lengths, hc, padding_value, max_seq_len):
        x = pack_padded_sequence(x, x_seq_lengths, batch_first=True, enforce_sorted=False)
        x, hc = self.lstm(x, hc)
        x, x_seq_lens = pad_packed_sequence(x, batch_first=True, padding_value=padding_value, total_length=max_seq_len)
        return x, x_seq_lens, hc


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers):
        super(Decoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, x_seq_lengths, hc, padding_value, max_seq_len):
        batch_size = x.shape[0]
        x = pack_padded_sequence(x, x_seq_lengths, batch_first=True, enforce_sorted=False)
        x, hc = self.lstm(x, hc)
        x, x_seq_lens = pad_packed_sequence(x, batch_first=True, padding_value=padding_value, total_length=max_seq_len)
        # x = x.contiguous()
        x = x.view(-1, self.hidden_size)
        x = self.linear(x)
        x = x.view(batch_size, -1, self.input_size)
        return x, x_seq_lens, hc


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        assert encoder.input_size == decoder.input_size
        assert encoder.hidden_size == decoder.hidden_size
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x_enc, x_dec, x_seq_lengths, hc_init, padding_value, max_seq_len):
        # print(x_enc.dtype, x_dec.dtype, x_seq_lengths.dtype, hc_init[0].dtype, hc_init[1].dtype)
        x_enc_out, _, hc_enc = self.encoder(x_enc, x_seq_lengths, hc_init, padding_value, max_seq_len)
        # print("x_enc.shape", x_enc.shape)
        # print("x_enc_out.shape", x_enc_out.shape)
        x_dec_out, _, hc_dec = self.decoder(x_dec, x_seq_lengths, hc_enc, padding_value, max_seq_len)
        return x_dec_out, hc_enc
    def get_embeddings_only(self, x_enc, x_seq_lengths, hc_init, padding_value, max_seq_len):
        _, _, hc_enc = self.encoder(x_enc, x_seq_lengths, hc_init, padding_value, max_seq_len)
        return hc_enc


def init_hidden(batch_size, hidden_size, num_rnn_layers, device):
    h = torch.zeros(num_rnn_layers, batch_size, hidden_size, device=device, dtype=torch.float32)
    c = torch.zeros(num_rnn_layers, batch_size, hidden_size, device=device, dtype=torch.float32)
    return (h, c)


def compute_loss(loss_function, x_pred, x_targ, x_seq_len):
    assert x_pred.shape == x_targ.shape
    
    mask = torch.ones_like(x_pred, dtype=int).to(x_pred.device)
    mask_seq_len = x_seq_len - 1  # As target sequence is one shorter.
    for idx, l in enumerate(mask_seq_len):
        mask[idx, l.item():, :] = 0.
    
    x_pred *= mask
    x_targ *= mask

    loss = loss_function(x_pred, x_targ)
    return loss
