"""
"""

import os

import numpy as np

import torch
import torch.optim as optim

from data.amsterdam import AmsterdamLoader, prepare_for_s2s_ae
from representations.ts_embedding import Encoder, Decoder, Seq2Seq, train_seq2seq_autoencoder, iterate_eval_set
from representations.ts_embedding import utils as s2s_utils


# ----------------------------------------------------------------------------------------------------------------------
# Set experiment settings here:

run_experiment = "dummy"  # Options: ("dummy", "amsterdam",)
models_dir = "./models/"
embeddings_dir = "./data/ts_embedding/"

# Dummy Data Experiment:
dummy_experiment_settings = {
    "n_samples_train": 1000,
    "n_samples_val": 500,
    "n_samples_test": 1000,
    "n_features": 2,
    "min_timesteps": 5,
    # --------------------
    "max_timesteps": 10,
    "pad_val": -999.,
    "eos_val": +777., 
    "data_gen_seed": 12345,
    "n_epochs": 1000,
    "batch_size": 128, 
    "hidden_size": 20,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    "model_name": "s2s_ae_dummy.pt",
    "embeddings_name": "dummy_embeddings.npy"
}

# Amsterdam Data Experiment:
amsterdam_experiment_settings = {
    "train_frac": 0.4,
    "val_frac": 0.2,
    "n_features": 70,
    # --------------------
    "max_timesteps": 100,
    "pad_val": -999.,
    "eos_val": +777., 
    "data_split_seed": 12345,
    "n_epochs": 100,
    "batch_size": 1024, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    "data_path": "data/amsterdam/combined_downsampled_longitudinal_data.csv",
    "model_name": "s2s_ae_amsterdam.pt",
    "embeddings_name": "amsterdam_embeddings.npy"
}

# ----------------------------------------------------------------------------------------------------------------------
# Utilities.

# Amsterdam data utilities.

def make_all_dataloaders(data_dict):
    dataloaders_dict = dict()
    for dataset_name, data_tensors in data_dict.items():
        dataset, dataloader = s2s_utils.make_dataloader(
            data_tensors=data_tensors, batch_size=amsterdam_experiment_settings["batch_size"], shuffle=False)
        dataloaders_dict[dataset_name] = dataloader
    return dataloaders_dict

def prepare_all_amsterdam_data(x_xlen_dict, device):
    data_dict = dict()
    for key in ("train", "val", "test"):
        x, x_len = x_xlen_dict[key]
        x_rev, x_rev_shifted = s2s_utils.rearrange_data(
            x, x_len, amsterdam_experiment_settings["pad_val"], amsterdam_experiment_settings["eos_val"])
        data_dict[key] = s2s_utils.data_to_tensors(
            x, x_len, x_rev, x_rev_shifted, float_type=torch.float32, device=device)
    return data_dict

# Dummy data utilities.

def generate_all_dummy_data(device):
    data_dict = dict()
    for key in ("train", "val", "test"):
        data_dict[key] = s2s_utils.generate_dummy_data(
            n_samples=dummy_experiment_settings[f"n_samples_{key}"], 
            min_timesteps=dummy_experiment_settings["min_timesteps"], 
            max_timesteps=dummy_experiment_settings["max_timesteps"], 
            n_features=dummy_experiment_settings["n_features"], 
            pad_val=dummy_experiment_settings["pad_val"], 
            eos_val=dummy_experiment_settings["eos_val"], 
            seed=dummy_experiment_settings["data_gen_seed"], 
            to_tensors=True,
            float_type=torch.float32, 
            device=device,
        )
    return data_dict


def main():
    if run_experiment == "dummy":
        active_experiment_settings = dummy_experiment_settings
    elif run_experiment == "amsterdam":
        active_experiment_settings = amsterdam_experiment_settings

    selected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if run_experiment == "dummy":
        data_dict = generate_all_dummy_data(device=selected_device)
        dataloaders_dict = make_all_dataloaders(data_dict)
    elif run_experiment == "amsterdam":
        data_path = os.path.abspath(active_experiment_settings["data_path"])
        amsterdam_loader = AmsterdamLoader(
            data_path=data_path,
            max_seq_len=amsterdam_experiment_settings["max_timesteps"],
            seed=amsterdam_experiment_settings["data_split_seed"],
            train_rate=amsterdam_experiment_settings["train_frac"],
            val_rate=amsterdam_experiment_settings["val_frac"],
            include_time=False,
            debug_data=False,
            pad_before=False,
            padding_fill=amsterdam_experiment_settings["pad_val"],
        )
        x_xlen_dict = prepare_for_s2s_ae(amsetrdam_loader=amsterdam_loader, force_refresh=True)
        data_dict = prepare_all_amsterdam_data(x_xlen_dict, device=selected_device)
        dataloaders_dict = make_all_dataloaders(data_dict)
    
    encoder = Encoder(
        input_size=active_experiment_settings["n_features"], 
        hidden_size=active_experiment_settings["hidden_size"], 
        num_rnn_layers=active_experiment_settings["num_rnn_layers"]
    )
    decoder = Decoder(
        input_size=active_experiment_settings["n_features"], 
        hidden_size=active_experiment_settings["hidden_size"], 
        num_rnn_layers=active_experiment_settings["num_rnn_layers"]
    )
    s2s = Seq2Seq(encoder=encoder, decoder=decoder)
    s2s.to(selected_device)

    opt = optim.Adam(s2s.parameters(), lr=active_experiment_settings["lr"])

    train_seq2seq_autoencoder(
        seq2seq=s2s, 
        optimizer=opt,
        train_dataloader=dataloaders_dict["train"],
        val_dataloader=dataloaders_dict["val"], 
        n_epochs=active_experiment_settings["n_epochs"], 
        batch_size=active_experiment_settings["batch_size"]
    )
    eval_loss = iterate_eval_set(seq2seq=s2s, dataloader=dataloaders_dict["test"])
    print(f"Ev.Ls.={eval_loss:.3f}")

    # Save model.
    model_filepath = os.path.join(os.path.abspath(models_dir), active_experiment_settings["model_name"])
    torch.save(s2s.state_dict(), model_filepath)

    # Save embeddings.
    embeddings_filepath = os.path.join(os.path.abspath(embeddings_dir), active_experiment_settings["embeddings_name"])
    embeddings = s2s_utils.get_embeddings(
        seq2seq=s2s, dataloaders_dict=dataloaders_dict, use_sets=("train", "val", "test"))
    np.save(embeddings_filepath, embeddings)
    print(f"Generated and saved embeddings of shape: {embeddings.shape}. File: {embeddings_filepath}.")


if __name__ == "__main__":
    main()
