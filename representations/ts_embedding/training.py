import numpy as np
import matplotlib.pyplot as plt
try:
    import IPython.display
except ImportError:
    print("IPython not found, ts_embedding > training live plot will not work.")

import torch
import torch.nn as nn
# import torch.optim as optim

from .seq2seq_autoencoder import init_hidden, compute_loss


loss_function = nn.MSELoss(reduction="none")


def iterate_eval_set(seq2seq, dataloader, padding_value, max_seq_len):
    epoch_test_loss = 0.
    
    seq2seq.eval()
    n_samples_test = 0
    with torch.no_grad():
        for iter_, (x, x_len, x_rev, x_rev_shift) in enumerate(dataloader):
            batch_size = x.shape[0]
            n_samples_test += batch_size

            hc_init = init_hidden(
                batch_size=batch_size, 
                hidden_size=seq2seq.encoder.hidden_size, 
                num_rnn_layers=seq2seq.encoder.num_rnn_layers, 
                device=x.device)

            x_dec_out, hc_repr = seq2seq(
                x_enc=x, 
                x_dec=x_rev, 
                x_seq_lengths=x_len, 
                hc_init=hc_init, 
                padding_value=padding_value, 
                max_seq_len=max_seq_len
            )
            
            loss_tensor = compute_loss(
                loss_function=loss_function, x_pred=x_dec_out, x_targ=x_rev_shift, x_seq_len=x_len)
            loss = loss_tensor.mean()
            epoch_test_loss += loss.item() * batch_size
        
    epoch_test_loss /= n_samples_test

    return epoch_test_loss


def train_seq2seq_autoencoder(
    seq2seq, 
    optimizer, 
    train_dataloader, 
    val_dataloader, 
    n_epochs, 
    batch_size, 
    padding_value, 
    max_seq_len,
    jupyter_live_plot_enabled=False
):
    
    train_losses, val_losses = np.full([n_epochs], np.nan), np.full([n_epochs], np.nan)
    x_axis = list(range(1, n_epochs + 1))
    
    for epoch in range(n_epochs):
        epoch_train_loss = 0.
        epoch_val_loss = 0.
        # print(f"Epoch {epoch}")
        
        seq2seq.train()
        n_samples_train = 0
        for iter_, (x, x_len, x_rev, x_rev_shift) in enumerate(train_dataloader):
            batch_size = x.shape[0]
            n_samples_train += batch_size
            
            optimizer.zero_grad()
            hc_init = init_hidden(
                batch_size=batch_size, 
                hidden_size=seq2seq.encoder.hidden_size, 
                num_rnn_layers=seq2seq.encoder.num_rnn_layers, 
                device=x.device)
            
            x_dec_out, hc_repr = seq2seq(
                x_enc=x, 
                x_dec=x_rev, 
                x_seq_lengths=x_len, 
                hc_init=hc_init, 
                padding_value=padding_value, 
                max_seq_len=max_seq_len
            )
            
            loss_tensor = compute_loss(
                loss_function=loss_function, x_pred=x_dec_out, x_targ=x_rev_shift, x_seq_len=x_len)
            loss = loss_tensor.mean()
            epoch_train_loss += loss.item() * batch_size
            
            loss.backward()
            optimizer.step()
        
        epoch_train_loss /= n_samples_train
        
        epoch_val_loss = iterate_eval_set(
            seq2seq=seq2seq, dataloader=val_dataloader, padding_value=padding_value, max_seq_len=max_seq_len)
        
        train_losses[epoch] = epoch_train_loss
        val_losses[epoch] = epoch_val_loss
        
        if jupyter_live_plot_enabled or (not jupyter_live_plot_enabled and epoch == n_epochs-1):
            # A live updating plot showing the training and validation over time (i.e. over epochs).
            plt.plot(x_axis, train_losses, label = "training loss")
            plt.plot(x_axis, val_losses, label = "validation loss")
            plt.title("Training Tracker")
            plt.legend()
            x_max = n_epochs
            y_max = np.nanmax(train_losses)
            plt.xlim(1, x_max)
            plt.ylim(0, y_max)
            if jupyter_live_plot_enabled:
                IPython.display.clear_output(wait=True)
            plt.show()
            plt.savefig("./training_log.png", dpi=300)
        
        print(f"Epoch {epoch}: Tr.Ls.={epoch_train_loss:.3f} Vl.Ls.={epoch_val_loss:.3f}")
