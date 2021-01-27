"""Time series embedding.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
import os
import copy

import numpy as np

import torch
import torch.optim as optim

from data.amsterdam import (
    AmsterdamLoader, 
    prepare_for_s2s_ae, 
    padding_mask_to_seq_lens, 
    convert_front_padding_to_back_padding
)
from representations.ts_embedding import Encoder, Decoder, Seq2Seq, train_seq2seq_autoencoder, iterate_eval_set
from representations.ts_embedding import utils as s2s_utils


# ----------------------------------------------------------------------------------------------------------------------
# Set experiment settings here:

# Options for `run_experiment`: 
#   - Learn embeddings:
#     "learn:dummy" 
#     "learn:amsterdam:combined_downsampled_subset" 
#     "learn:amsterdam:test_subset"
#   - Apply existing embeddings:
#     "apply:amsterdam:hns_competition_data"
run_experiment = "apply:amsterdam:hns_competition_data"

models_dir = "./models/"
embeddings_dir = "./data/ts_embedding/"
experiment_settings = dict()

# Dummy Data Learn Autoencoder Experiment:
experiment_settings["learn:dummy"] = {
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

# Amsterdam Data Learn Autoencoder Experiments:
# - "learn:amsterdam:combined_downsampled_subset"
experiment_settings["learn:amsterdam:combined_downsampled_subset"] = {
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
    "model_name": "s2s_ae_amsterdam_comb.pt",
    "embeddings_name": "amsterdam_embeddings_comb.npy"
}
# - "learn:amsterdam:test_subset"
experiment_settings["learn:amsterdam:test_subset"] = \
    copy.deepcopy(experiment_settings["learn:amsterdam:combined_downsampled_subset"])
experiment_settings["learn:amsterdam:test_subset"]["data_path"] = "data/amsterdam/test_longitudinal_data.csv"
experiment_settings["learn:amsterdam:test_subset"]["model_name"] = "s2s_ae_amsterdam_test.pt"
experiment_settings["learn:amsterdam:test_subset"]["embeddings_name"] = "amsterdam_embeddings_test.npy"

# Hide-and-seek Competition Apply Autoencoder Experiment:
experiment_settings["apply:amsterdam:hns_competition_data"] = {
    "gen_data_path": "./data/ts_generated/hns_comp/",
    "hiders_list": [
        "csetraynor", 
        "Atrin", 
        "saeedsa",
        "tuscan-chicken-wrap",
        "yingruiz",
        "hns_baseline_timegan",
        "akashdeepsingh",
        "wangzq312",
        "flynngo",
        "lumip",
        "wangz10",
        "SatoshiHasegawa",
        "yingjialin",
        "hamada",
        "jilljenn",
        "hns_baseline_add_noise",
    ],
    "data_file_name": "data.npz",
    "pad_val": -999.,
    "max_timesteps": 100,
    # --------------------
    "model_path": "./models/s2s_ae_amsterdam_test.pt",
    "batch_size": 1024, 
    "n_features": 70,
    "hidden_size": 70,
    "num_rnn_layers": 2,
    # --------------------
    "embeddings_subdir": "hns_comp",
    "embeddings_name": "hns_embeddings_<uname>.npy",
    # --------------------
    # NOTE: The exact same train-test split applied to `test_longitudinal_data.csv` as in the competition, 
    # the test subset is saved as original data.
    "original_data_path":  "data/amsterdam/test_longitudinal_data.csv",
    "original_data_train_rate": 0.5,
    "original_data_val_rate": 0.,
    "original_data_split_seed": 12345,
    "original_data_embeddings_name": "hns_embeddings_ORIGINAL_DATA.npy",
}

# ----------------------------------------------------------------------------------------------------------------------
# Utilities.

# Amsterdam data utilities.

def make_all_dataloaders(data_dict, exp_settings):
    dataloaders_dict = dict()
    for dataset_name, data_tensors in data_dict.items():
        dataset, dataloader = s2s_utils.make_dataloader(
            data_tensors=data_tensors, batch_size=exp_settings["batch_size"], shuffle=False)
        dataloaders_dict[dataset_name] = dataloader
    return dataloaders_dict

def prepare_all_amsterdam_data(x_xlen_dict, device, exp_settings):
    data_dict = dict()
    for key in ("train", "val", "test"):
        x, x_len = x_xlen_dict[key]
        x_rev, x_rev_shifted = s2s_utils.rearrange_data(
            x, x_len, exp_settings["pad_val"], exp_settings["eos_val"])
        data_dict[key] = s2s_utils.data_to_tensors(
            x, x_len, x_rev, x_rev_shifted, float_type=torch.float32, device=device)
    return data_dict

# Dummy data utilities.

def generate_all_dummy_data(device):
    dummy_exp_stgs = experiment_settings["learn:dummy"]
    data_dict = dict()
    for key in ("train", "val", "test"):
        data_dict[key] = s2s_utils.generate_dummy_data(
            n_samples=dummy_exp_stgs[f"n_samples_{key}"], 
            min_timesteps=dummy_exp_stgs["min_timesteps"], 
            max_timesteps=dummy_exp_stgs["max_timesteps"], 
            n_features=dummy_exp_stgs["n_features"], 
            pad_val=dummy_exp_stgs["pad_val"], 
            eos_val=dummy_exp_stgs["eos_val"], 
            seed=dummy_exp_stgs["data_gen_seed"], 
            to_tensors=True,
            float_type=torch.float32, 
            device=device,
        )
    return data_dict

# Hide-and-seek competition data utilities.

def _check_padding_mask_integrity(padding_mask):
    for idx_sample, sample in enumerate(padding_mask):
        use_feature = sample[:, 0]  # Assume all features have the same mask, as they should.
        if use_feature.astype(int).sum() == 0:
            pass  # Good (no padding in the sample, i.e. max_seq_len).
        else:
            # Check (a) pre-padded, and (b) continuous.
            assert use_feature[0] == True
            for idx in range(0, len(use_feature)-1):
                curr = use_feature[idx]
                nex = use_feature[idx+1]
                assert (curr, nex) != (False, True)


def prepare_hns_gen_data(hider_name, exp_settings):

    data = np.load(os.path.join(exp_settings["gen_data_path"], hider_name, exp_settings["data_file_name"]))
    generated_data = data["generated_data"]
    padding_mask = data["padding_mask"]

    # Check padding mask integrity
    if padding_mask.shape != (0,):
        assert padding_mask.shape == (7695, 100, 71)
        _check_padding_mask_integrity(padding_mask)
        note = "- has a padding mask."
    else:
        note = "- NO padding mask."
    print(f"hider '{hider_name}' padding_mask checked {note}")

    if padding_mask.shape != (0,):
        # Has a padding mask case.
        seq_lens = padding_mask_to_seq_lens(padding_mask=padding_mask)
        generated_data = convert_front_padding_to_back_padding(
            data=generated_data, 
            seq_lens=seq_lens, 
            pad_val=exp_settings["pad_val"]
        )
    else:
        # No padding mask case, all seq_lens are max length.
        seq_lens = np.array([generated_data.shape[1]] * generated_data.shape[0]).astype(int)

    # Remove the time feature.
    generated_data = generated_data[:, :, 1:]
    if padding_mask.shape != (0,):
        padding_mask = padding_mask[:, :, 1:]
    else:
        padding_mask = None

    # Impute in case any generated data had nans.
    n_nan = np.isnan(generated_data).astype(int).sum()
    if n_nan > 0:
        generated_data = AmsterdamLoader.impute_only(
            data=generated_data, padding_mask=padding_mask, padding_fill=exp_settings["pad_val"])
        n_nan_after = np.isnan(generated_data).astype(int).sum()
        assert n_nan_after == 0
        print(f"{n_nan} nan values in the generated data were imputed.")
    
    return generated_data, seq_lens


def get_hns_dataloader(x, x_len, device, exp_settings):
    X, X_len = s2s_utils.inference_data_to_tensors(x, x_len, float_type=torch.float32, device=device)
    dataset, dataloader = s2s_utils.make_dataloader(
            data_tensors=(X, X_len), 
            batch_size=exp_settings["batch_size"], 
            shuffle=False)
    return dataset, dataloader


def main():

    exp_settings = experiment_settings[run_experiment]
    selected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Autoencoder learning experiments.
    if "learn:" in run_experiment:

        if run_experiment == "learn:dummy":
            data_dict = generate_all_dummy_data(device=selected_device)
            dataloaders_dict = make_all_dataloaders(data_dict, exp_settings=exp_settings)
        elif (
            run_experiment == "learn:amsterdam:combined_downsampled_subset" or 
            run_experiment == "learn:amsterdam:test_subset"
        ):
            data_path = os.path.abspath(exp_settings["data_path"])
            amsterdam_loader = AmsterdamLoader(
                data_path=data_path,
                max_seq_len=exp_settings["max_timesteps"],
                seed=exp_settings["data_split_seed"],
                train_rate=exp_settings["train_frac"],
                val_rate=exp_settings["val_frac"],
                include_time=False,
                debug_data=False,
                pad_before=False,
                padding_fill=exp_settings["pad_val"],
            )
            x_xlen_dict = prepare_for_s2s_ae(amsterdam_loader=amsterdam_loader, force_refresh=True)
            data_dict = prepare_all_amsterdam_data(x_xlen_dict, device=selected_device, exp_settings=exp_settings)
            dataloaders_dict = make_all_dataloaders(data_dict, exp_settings=exp_settings)
        
        encoder = Encoder(
            input_size=exp_settings["n_features"], 
            hidden_size=exp_settings["hidden_size"], 
            num_rnn_layers=exp_settings["num_rnn_layers"]
        )
        decoder = Decoder(
            input_size=exp_settings["n_features"], 
            hidden_size=exp_settings["hidden_size"], 
            num_rnn_layers=exp_settings["num_rnn_layers"]
        )
        s2s = Seq2Seq(encoder=encoder, decoder=decoder)
        s2s.to(selected_device)

        opt = optim.Adam(s2s.parameters(), lr=exp_settings["lr"])

        train_seq2seq_autoencoder(
            seq2seq=s2s, 
            optimizer=opt,
            train_dataloader=dataloaders_dict["train"],
            val_dataloader=dataloaders_dict["val"], 
            n_epochs=exp_settings["n_epochs"], 
            batch_size=exp_settings["batch_size"]
        )
        eval_loss = iterate_eval_set(seq2seq=s2s, dataloader=dataloaders_dict["test"])
        print(f"Ev.Ls.={eval_loss:.3f}")

        # Save model.
        model_filepath = os.path.join(os.path.abspath(models_dir), exp_settings["model_name"])
        torch.save(s2s.state_dict(), model_filepath)

        # Save embeddings.
        embeddings_filepath = os.path.join(os.path.abspath(embeddings_dir), exp_settings["embeddings_name"])
        embeddings = s2s_utils.get_embeddings(
            seq2seq=s2s, 
            dataloaders=(dataloaders_dict["train"], dataloaders_dict["val"], dataloaders_dict["test"])
        )
        np.save(embeddings_filepath, embeddings)
        print(f"Generated and saved embeddings of shape: {embeddings.shape}. File: {embeddings_filepath}.")
    
    # Autoencoder application experiments.
    elif "apply:" in run_experiment:
        
        if run_experiment == "apply:amsterdam:hns_competition_data":
            
            # Load up the model.
            encoder = Encoder(
                input_size=exp_settings["n_features"], 
                hidden_size=exp_settings["hidden_size"], 
                num_rnn_layers=exp_settings["num_rnn_layers"]
            )
            decoder = Decoder(
                input_size=exp_settings["n_features"], 
                hidden_size=exp_settings["hidden_size"], 
                num_rnn_layers=exp_settings["num_rnn_layers"]
            )
            s2s = Seq2Seq(encoder=encoder, decoder=decoder)
            s2s.to(selected_device)
            s2s.load_state_dict(torch.load(exp_settings["model_path"]))

            for hider_name in exp_settings["hiders_list"]:
                
                # Prepare the data.
                generated_data, seq_lens = prepare_hns_gen_data(hider_name=hider_name, exp_settings=exp_settings)
                dataset, dataloader = get_hns_dataloader(
                    x=generated_data, 
                    x_len=seq_lens, 
                    device=selected_device, 
                    exp_settings=exp_settings)

                # Save embeddings.
                embeddings_filepath = os.path.join(
                    os.path.abspath(embeddings_dir), 
                    exp_settings["embeddings_subdir"],
                    exp_settings["embeddings_name"].replace("<uname>", hider_name)
                )
                embeddings = s2s_utils.get_embeddings(seq2seq=s2s, dataloaders=(dataloader,))
                np.save(embeddings_filepath, embeddings)
                n_nan = np.isnan(embeddings).astype(int).sum()
                assert n_nan == 0
                print(f"Generated and saved embeddings of shape: {embeddings.shape}. File: {embeddings_filepath}.")
            
            # Embed also original data.
            data_path = os.path.abspath(exp_settings["original_data_path"])
            amsterdam_loader = AmsterdamLoader(
                data_path=data_path,
                max_seq_len=exp_settings["max_timesteps"],
                seed=exp_settings["original_data_split_seed"],
                train_rate=exp_settings["original_data_train_rate"],
                val_rate=exp_settings["original_data_val_rate"],
                include_time=False,
                debug_data=False,
                pad_before=False,
                padding_fill=exp_settings["pad_val"],
            )
            x_xlen_dict = prepare_for_s2s_ae(amsterdam_loader=amsterdam_loader, force_refresh=True)
            x, x_len = x_xlen_dict["test"]
            dataset, dataloader = get_hns_dataloader(
                x=x, x_len=x_len, device=selected_device,exp_settings=exp_settings)
            embeddings_filepath = os.path.join(
                os.path.abspath(embeddings_dir), 
                exp_settings["embeddings_subdir"],
                exp_settings["original_data_embeddings_name"]
            )
            embeddings = s2s_utils.get_embeddings(seq2seq=s2s, dataloaders=(dataloader,))
            np.save(embeddings_filepath, embeddings)
            print(f"Generated and saved embeddings of shape: {embeddings.shape}. File: {embeddings_filepath}.")


if __name__ == "__main__":
    main()
