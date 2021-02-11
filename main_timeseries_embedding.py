"""Time series embedding.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
import os

import numpy as np

import torch
import torch.optim as optim

from data import amsterdam
from data import googlestock
from data import snp500
from utils import prepare_amsterdam
from representations.ts_embedding import Encoder, Decoder, Seq2Seq, train_seq2seq_autoencoder, iterate_eval_set
from representations.ts_embedding import utils as s2s_utils


# ----------------------------------------------------------------------------------------------------------------------
# Set experiment settings here:

# Options for `run_experiment`: 
#   - Learn embeddings:
#     "learn:dummy" 
#     "learn:googlestock"
#     "learn:snp500" 
#     "learn:amsterdam:combined_downsampled_subset" 
#     "learn:amsterdam:test_subset"
#     "learn:amsterdam:hns_subset"
#   - Apply existing embeddings:
#     "apply:amsterdam:hns_competition_data"
#     "apply:amsterdam:combined_downsampled_subset"
run_experiment = "learn:amsterdam:combined_downsampled_subset"

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

# Google Stock Learn Autoencoder Experiment:
experiment_settings["learn:googlestock"] = {
    "train_frac": 1./3.,
    "val_frac": 1./3.,
    "split_order": {"train": 2, "test": 3, "val": 1},
    "n_features": 5,
    # --------------------
    "include_time": False,
    "max_timesteps": None,  # Calculated automatically.
    "pad_val": -999.,
    "eos_val": -777.,  
    # NOTE: No "data_split_seed", as data is split along the time axis, no shuffling.
    # --------------------
    "n_epochs": 500,
    "batch_size": 1, 
    "hidden_size": 20,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    "data_path_trainfile": "./data/googlestock/Google_Stock_Price_Train.csv",
    "data_path_testfile": "./data/googlestock/Google_Stock_Price_Test.csv",
    "model_name": "s2s_ae_googlestock.pt",
    "embeddings_name": "googlestock_embeddings.npy"
}

# S&P 500 Learn Autoencoder Experiment:
experiment_settings["learn:snp500"] = {
    "train_frac": 0.4,
    "val_frac": 0.2,
    "n_features": 5,
    # --------------------
    "include_time": False,
    "max_timesteps": 1259,
    "pad_val": -999.,
    "eos_val": -777.,
    "data_split_seed": 12345,
    "data_loading_force_refresh": True,
    # --------------------
    "n_epochs": 100,
    "batch_size": 256, 
    "hidden_size": 20,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    "data_path": "./data/snp500/all_stocks_5yr.csv",
    "npz_cache_filepath": "./data/snp500/snp500.npz",
    "model_name": "s2s_ae_snp500.pt",
    "embeddings_name": "snp500_embeddings.npy"
}


# Amsterdam Data Learn Autoencoder Experiments:
# - "learn:amsterdam:combined_downsampled_subset"
# NOTE: requires first running main() in ./data/amsterdam/data_scripts.py 
# to make combined_downsampled5000_longitudinal_data.csv.
_use_amsterdam_comb_version = "5000"  # Options: ("1000", "5000")
experiment_settings["learn:amsterdam:combined_downsampled_subset"] = {
    "train_frac": 0.4,
    "val_frac": 0.2,
    "n_features": 70,
    # --------------------
    "include_time": False,
    "max_timesteps": [10, 100, 1000],
    "pad_val": -999.,
    "eos_val": +777., 
    "data_split_seed": 12345,
    "data_loading_force_refresh": True,
    # --------------------
    "n_epochs": [1000, 200, 50],
    "batch_size": 1024, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    "data_path": f"./data/amsterdam/combined_downsampled{_use_amsterdam_comb_version}_longitudinal_data.csv",
    "model_name": f"s2s_ae_amsterdam_comb{_use_amsterdam_comb_version}_<max_ts>.pt",
    "embeddings_name": f"amsterdam_embeddings_comb{_use_amsterdam_comb_version}_<max_ts>.npy"
}
# - "learn:amsterdam:test_subset"
experiment_settings["learn:amsterdam:test_subset"] = {
    "train_frac": 0.4,
    "val_frac": 0.2,
    "n_features": 70,
    # --------------------
    "include_time": False,
    "max_timesteps": 100,
    "pad_val": -999.,
    "eos_val": +777., 
    "data_split_seed": 12345,
    "data_loading_force_refresh": True,
    # --------------------
    "n_epochs": 100,
    "batch_size": 1024, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    "data_path": "./data/amsterdam/test_longitudinal_data.csv",
    "model_name": "s2s_ae_amsterdam_test.pt",
    "embeddings_name": "amsterdam_embeddings_test.npy"
}
# - "learn:amsterdam:hns_subset"
experiment_settings["learn:amsterdam:hns_subset"] = {
    "train_frac": 0.4,
    "val_frac": 0.2,
    "n_features": 70,
    # --------------------
    "data_load": {
        "force_refresh": True,
        "include_time": True,
        "max_timesteps": 100,
        "pad_val": -1.,
        "train_frac": 0.5,
        "val_frac": 0.,
        "data_split_seed": 12345,
    },
    # --------------------
    "max_timesteps": 100,
    "pad_val": -1.,
    "eos_val": +2., 
    "data_split_seed": 22222,
    # --------------------
    "n_epochs": 100,
    "batch_size": 1024, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    "data_path": "./data/amsterdam/hns_test_longitudinal_data.csv",  # NOTE: copy of test_longitudinal_data.csv
    "model_name": "s2s_ae_amsterdam_hns.pt",
    "embeddings_name": "amsterdam_embeddings_hns.npy"
}

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
        # Augmented (10x hamada generated data):
        "AUDITING",
    ],
    "data_file_name": "data.npz",
    "pad_val": experiment_settings["learn:amsterdam:hns_subset"]["pad_val"],
    "eos_val": experiment_settings["learn:amsterdam:hns_subset"]["eos_val"],
    "max_timesteps": experiment_settings["learn:amsterdam:hns_subset"]["max_timesteps"],
    # --------------------
    "model_path": "./models/s2s_ae_amsterdam_hns.pt",
    "batch_size": experiment_settings["learn:amsterdam:hns_subset"]["batch_size"], 
    "n_features": experiment_settings["learn:amsterdam:hns_subset"]["n_features"],
    "hidden_size": experiment_settings["learn:amsterdam:hns_subset"]["hidden_size"],
    "num_rnn_layers": experiment_settings["learn:amsterdam:hns_subset"]["num_rnn_layers"],
    # --------------------
    "embeddings_subdir": "hns_comp",
    "embeddings_name": "hns_embeddings_<uname>.npy",
    # --------------------
    "proc_cached_dir": "./data/ts_generated/hns_comp_proc_cached",
    "load_from_proc_cached": False,
}

_use_seq_len_apply_amsterdam_comb = 100
experiment_settings["apply:amsterdam:combined_downsampled_subset"] = {
    "gen_data_path": "./data/ts_generated/",
    "generated_data_name": "amsterdam_combined_downsampled_subset_<model_name>.npy",
    "models_list": [
        "rgan",
    ],
    # --------------------
    "include_time": False,
    "n_features": 70,
    "max_timesteps": _use_seq_len_apply_amsterdam_comb,
    "pad_val": -999.,
    "eos_val": +777., 
    # --------------------
    "batch_size": 1024, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    # --------------------
    "model_path": f"./models/s2s_ae_amsterdam_comb_{_use_seq_len_apply_amsterdam_comb}.pt",
    # --------------------
    "embeddings_name": f"amsterdam_embeddings_comb_{_use_seq_len_apply_amsterdam_comb}_<model_name>.npy",
    # --------------------
}

# ----------------------------------------------------------------------------------------------------------------------
# Utilities.

# General utilities.
def make_all_dataloaders(data_dict, batch_size):
    dataloaders_dict = dict()
    for dataset_name, data_tensors in data_dict.items():
        dataset, dataloader = s2s_utils.make_dataloader(
            data_tensors=data_tensors, batch_size=batch_size, shuffle=False)
        dataloaders_dict[dataset_name] = dataloader
    return dataloaders_dict


def prep_datasets_for_s2s_ae_training(x_xlen_dict, device, pad_val, eos_val):
    data_dict = dict()
    for key in x_xlen_dict.keys():
        x, x_len = x_xlen_dict[key]
        x_rev, x_rev_shifted = s2s_utils.rearrange_data(x, x_len, pad_val, eos_val)
        data_dict[key] = s2s_utils.data_to_tensors(
            x, x_len, x_rev, x_rev_shifted, float_type=torch.float32, device=device)
    return data_dict


def get_inference_dataloader(x, x_len, device, exp_settings):
    x_rev, x_rev_shifted = s2s_utils.rearrange_data(
            x, x_len, exp_settings["pad_val"], exp_settings["eos_val"])

    X, X_len, X_rev, X_rev_shifted = s2s_utils.data_to_tensors(
        x, x_len, x_rev, x_rev_shifted, float_type=torch.float32, device=device)

    dataset, dataloader = s2s_utils.make_dataloader(
            data_tensors=(X, X_len, X_rev, X_rev_shifted), 
            batch_size=exp_settings["batch_size"], 
            shuffle=False)

    return dataset, dataloader


# Google stock data utilities.

def add_sample_dim(*arrays):
    results = []
    for arr in arrays:
        assert len(arr.shape) == 2
        results.append(np.expand_dims(arr, axis=0))
    return tuple(results)


def get_fixed_seq_lens(*arrays):
    results = []
    for arr in arrays:
        assert len(arr.shape) == 3
        results.append(np.full((arr.shape[0],), arr.shape[1]))
    return tuple(results)


def pad_to_max_seq_len(arrays, max_seq_len, pad_val):
    results = []
    for arr in arrays:
        assert len(arr.shape) == 3
        arr_ = np.full((arr.shape[0], max_seq_len, arr.shape[2]), pad_val)
        for idx in range(arr.shape[0]):
            arr_[idx, :arr.shape[1], :] = arr[idx, :, :]
        results.append(arr_)
    return tuple(results)


# Amsterdam data utilities.

def prepare_amsterdam_and_split(amsterdam_loader, settings):
    imputed_processed_data, seq_lens, (train_idx, val_idx, test_idx) = prepare_amsterdam(amsterdam_loader, settings)
    data = {
        "train": (imputed_processed_data[train_idx], seq_lens[train_idx]),
        "val": (imputed_processed_data[val_idx], seq_lens[val_idx]),
        "test": (imputed_processed_data[test_idx], seq_lens[test_idx]),
    }
    return data


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


def _coerce_hns_data_to_s2s_ae_format(data, padding_mask, pad_val):
    if padding_mask.shape != (0,):
        # Has a padding mask case.
        seq_lens = amsterdam.padding_mask_to_seq_lens(padding_mask=padding_mask)
        data = amsterdam.convert_front_padding_to_back_padding(data=data, seq_lens=seq_lens, pad_val=pad_val)
    else:
        # No padding mask case, all seq_lens are max length.
        seq_lens = np.array([data.shape[1]] * data.shape[0]).astype(int)

    # Remove the time feature.
    data = data[:, :, 1:]
    if padding_mask.shape != (0,):
        padding_mask = padding_mask[:, :, 1:]
    else:
        padding_mask = None
    
    return data, padding_mask, seq_lens


def prepare_for_s2s_ae_hns(amsterdam_loader, settings):
    
    # First obtain the same data subset as in H&S competition.
    assert amsterdam_loader.pad_before == True
    raw_data, padding_mask, (train_idx_, _, _) = \
        amsterdam_loader.load_reshape_split_data(force_refresh=settings["data_load"]["force_refresh"])
    hns_raw_data, hns_padding_mask = raw_data[train_idx_], padding_mask[train_idx_]  
    # ^ The dataset used in H&S competition.

    # Now get the split for autoencoder training:
    _, (train_idx, val_idx, test_idx) = amsterdam.data_division(
        hns_raw_data, 
        seed=settings["data_split_seed"], 
        divide_rates=[
            settings["train_frac"], 
            settings["val_frac"], 
            1 - settings["train_frac"] - settings["val_frac"]
        ]
    )

    # Preprocess.
    _, hns_imputed_processed_data = amsterdam.preprocess_data(
        data=hns_raw_data, 
        padding_mask=hns_padding_mask, 
        padding_fill=settings["data_load"]["pad_val"],
        time_feature_included=settings["data_load"]["include_time"],
    )
    # Coerce to Seq2Seq autoencoder input format.
    hns_final_data, hns_final_padding_mask, hns_seq_lens = _coerce_hns_data_to_s2s_ae_format(
        data=hns_imputed_processed_data, 
        padding_mask=hns_padding_mask, 
        pad_val=settings["data_load"]["pad_val"]
    )
    
    data = {
        "train": (hns_final_data[train_idx], hns_seq_lens[train_idx]),
        "val": (hns_final_data[val_idx], hns_seq_lens[val_idx]),
        "test": (hns_final_data[test_idx], hns_seq_lens[test_idx]),
    }
    return data


def prepare_hns_gen_data(hider_name, exp_settings):

    data = np.load(os.path.join(exp_settings["gen_data_path"], hider_name, exp_settings["data_file_name"]))
    generated_data = data["generated_data"]
    padding_mask = data["padding_mask"]

    # Check padding mask integrity.
    if padding_mask.shape != (0,):
        if hider_name != "AUDITING":
            assert padding_mask.shape == (7695, 100, 71)
        else:
            assert padding_mask.shape == (76950, 100, 71)
        _check_padding_mask_integrity(padding_mask)
        note = "- has a padding mask."
    else:
        note = "- NO padding mask."
    print(f"hider '{hider_name}' padding_mask checked {note}")

    # Preprocess.
    _, generated_data = amsterdam.preprocess_data(
        data=generated_data, 
        padding_mask=padding_mask if padding_mask.shape != (0,) else None, 
        padding_fill=exp_settings["pad_val"],
        time_feature_included=True,
    )
    # Coerce to Seq2Seq autoencoder input format.
    generated_data, padding_mask, seq_lens = _coerce_hns_data_to_s2s_ae_format(
        data=generated_data, padding_mask=padding_mask, pad_val=exp_settings["pad_val"])

    return generated_data, seq_lens

# ----------------------------------------------------------------------------------------------------------------------


def main():

    exp_settings = experiment_settings[run_experiment]
    selected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Autoencoder learning experiments.
    if "learn:" in run_experiment:

        if run_experiment == "learn:dummy":
            data_dict = generate_all_dummy_data(device=selected_device)
            dataloaders_dict = make_all_dataloaders(data_dict, batch_size=exp_settings["batch_size"])
        
        elif run_experiment == "learn:googlestock":
            # Load and split:
            d_full = googlestock.load_stock_data(
                train_path=exp_settings["data_path_trainfile"], 
                test_path=exp_settings["data_path_testfile"], 
                normalize=True, 
                time=False
            )
            d_train, d_val, d_test = googlestock.split_stock_data(
                d_full, 
                frac_train=exp_settings["train_frac"], 
                frac_val=exp_settings["val_frac"], 
                split_order=exp_settings["split_order"]
            )
            # Dynamically compute max_seq_len:
            max_timesteps = max((d_train.shape[0], d_val.shape[0], d_test.shape[0]))
            total_timesteps = d_full.shape[0]
            exp_settings["max_timesteps"] = max_timesteps
            print(f"`max_timesteps` computed: {max_timesteps}")
            # Preprocess for autoencoder:
            d_train, d_val, d_test, d_full = \
                add_sample_dim(d_train, d_val, d_test, d_full)  # pylint: disable=unbalanced-tuple-unpacking
            d_train_len, d_val_len, d_test_len, d_full_len = \
                get_fixed_seq_lens(d_train, d_val, d_test, d_full)  # pylint: disable=unbalanced-tuple-unpacking
            d_train, d_val, d_test = pad_to_max_seq_len(  # pylint: disable=unbalanced-tuple-unpacking
                arrays=(d_train, d_val, d_test), 
                max_seq_len=exp_settings["max_timesteps"], 
                pad_val=exp_settings["pad_val"]
            )
            x_xlen_dict = {
                "train": (d_train, d_train_len),
                "val": (d_val, d_val_len),
                "test": (d_test, d_test_len),
                "full": (d_full, d_full_len)
            }
            data_dict = prep_datasets_for_s2s_ae_training(
                x_xlen_dict=x_xlen_dict, 
                device=selected_device, 
                pad_val=exp_settings["pad_val"], 
                eos_val=exp_settings["eos_val"]
            )
            dataloaders_dict = make_all_dataloaders(data_dict=data_dict, batch_size=exp_settings["batch_size"])

        elif run_experiment == "learn:snp500":
            processed_data, seq_lens, _ = snp500.load_snp_data(
                data_path=exp_settings["data_path"], 
                npz_cache_filepath=exp_settings["npz_cache_filepath"], 
                padding_value=exp_settings["pad_val"], 
                normalize=True, 
                include_time=exp_settings["include_time"], 
                force_refresh=exp_settings["data_loading_force_refresh"], 
            )
            train_x_xlen, val_x_xlen, test_x_xlen = snp500.split_snp_data(
                data=processed_data, 
                seq_lens=seq_lens, 
                frac_train=exp_settings["train_frac"], 
                frac_val=exp_settings["val_frac"], 
                seed=exp_settings["data_split_seed"]
            )
            x_xlen_dict = {"train": train_x_xlen, "val": val_x_xlen, "test": test_x_xlen}
            data_dict = prep_datasets_for_s2s_ae_training(
                x_xlen_dict=x_xlen_dict, 
                device=selected_device, 
                pad_val=exp_settings["pad_val"], 
                eos_val=exp_settings["eos_val"]
            )
            dataloaders_dict = make_all_dataloaders(data_dict=data_dict, batch_size=exp_settings["batch_size"])

        elif run_experiment == "learn:amsterdam:test_subset":
            amsterdam_loader = amsterdam.AmsterdamLoader(
                data_path=os.path.abspath(exp_settings["data_path"]),
                max_seq_len=exp_settings["max_timesteps"],
                seed=exp_settings["data_split_seed"],
                train_rate=exp_settings["train_frac"],
                val_rate=exp_settings["val_frac"],
                include_time=exp_settings["include_time"],
                debug_data=False,
                pad_before=False,
                padding_fill=exp_settings["pad_val"],
            )
            x_xlen_dict = prepare_amsterdam_and_split(
                amsterdam_loader=amsterdam_loader, 
                settings=exp_settings,
            )
            data_dict = prep_datasets_for_s2s_ae_training(
                x_xlen_dict, 
                device=selected_device, 
                pad_val=exp_settings["pad_val"],
                eos_val=exp_settings["eos_val"],
            )
            dataloaders_dict = make_all_dataloaders(data_dict, batch_size=exp_settings["batch_size"])
        
        elif run_experiment == "learn:amsterdam:combined_downsampled_subset":
            dataloaders_dict = []
            for max_timesteps in exp_settings["max_timesteps"]:
                amsterdam_loader = amsterdam.AmsterdamLoader(
                    data_path=os.path.abspath(exp_settings["data_path"]),
                    max_seq_len=max_timesteps,
                    seed=exp_settings["data_split_seed"],
                    train_rate=exp_settings["train_frac"],
                    val_rate=exp_settings["val_frac"],
                    include_time=exp_settings["include_time"],
                    debug_data=False,
                    pad_before=False,
                    padding_fill=exp_settings["pad_val"],
                )
                x_xlen_dict = prepare_amsterdam_and_split(
                    amsterdam_loader=amsterdam_loader, 
                    settings=exp_settings,
                )
                data_dict = prep_datasets_for_s2s_ae_training(
                    x_xlen_dict, 
                    device=selected_device, 
                    pad_val=exp_settings["pad_val"],
                    eos_val=exp_settings["eos_val"],
                )
                dataloaders_dict.append( make_all_dataloaders(data_dict, batch_size=exp_settings["batch_size"]))
        
        elif run_experiment == "learn:amsterdam:hns_subset":
            # This loader matches the H&S competition data loading settings.
            amsterdam_loader = amsterdam.AmsterdamLoader(
                data_path=os.path.abspath(exp_settings["data_path"]),
                max_seq_len=exp_settings["data_load"]["max_timesteps"],
                seed=exp_settings["data_load"]["data_split_seed"],
                train_rate=exp_settings["data_load"]["train_frac"],
                val_rate=exp_settings["data_load"]["val_frac"],
                include_time=exp_settings["data_load"]["include_time"],
                debug_data=False,
                pad_before=True,  # NOTE: This matches the H&S competition setup.
                padding_fill=exp_settings["data_load"]["pad_val"],
            )
            x_xlen_dict = prepare_for_s2s_ae_hns(
                amsterdam_loader=amsterdam_loader, 
                settings=exp_settings,
            )
            data_dict = prep_datasets_for_s2s_ae_training(
                x_xlen_dict, 
                device=selected_device, 
                pad_val=exp_settings["pad_val"],
                eos_val=exp_settings["eos_val"],
            )
            dataloaders_dict = make_all_dataloaders(data_dict, batch_size=exp_settings["batch_size"])
        
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

        if run_experiment == "learn:amsterdam:combined_downsampled_subset":
            # Multiple training runs.
            
            n_epochs_list = exp_settings["n_epochs"]
            max_timesteps_list = exp_settings["max_timesteps"]
            
            assert len(dataloaders_dict) == len(n_epochs_list)
            
            model_name_list = \
                [exp_settings["model_name"].replace("<max_ts>", str(max_ts)) for max_ts in max_timesteps_list]
            embeddings_name_list = \
                [exp_settings["embeddings_name"].replace("<max_ts>", str(max_ts)) for max_ts in max_timesteps_list]
        
        else:
            # Single training run.
            n_epochs_list = [exp_settings["n_epochs"]]
            max_timesteps_list = [exp_settings["max_timesteps"]]
            dataloaders_dict = [dataloaders_dict]
            model_name_list = [exp_settings["model_name"]]
            embeddings_name_list = [exp_settings["embeddings_name"]]
        
        assert (
            len(n_epochs_list) == 
            len(max_timesteps_list) == 
            len(dataloaders_dict) == 
            len(model_name_list) == 
            len(embeddings_name_list)
        )

        for idx in range(len(n_epochs_list)):
            train_seq2seq_autoencoder(
                seq2seq=s2s, 
                optimizer=opt,
                train_dataloader=dataloaders_dict[idx]["train"],
                val_dataloader=dataloaders_dict[idx]["val"], 
                n_epochs=n_epochs_list[idx], 
                batch_size=exp_settings["batch_size"],
                padding_value=exp_settings["pad_val"],
                max_seq_len=max_timesteps_list[idx],
            )
            eval_loss = iterate_eval_set(
                seq2seq=s2s, 
                dataloader=dataloaders_dict[idx]["test"],
                padding_value=exp_settings["pad_val"],
                max_seq_len=max_timesteps_list[idx],
            )
            print(f"Ev.Ls.={eval_loss:.3f}")

            # Save model.
            model_filepath = os.path.join(os.path.abspath(models_dir), model_name_list[idx])
            torch.save(s2s.state_dict(), model_filepath)

            # Save embeddings.
            if run_experiment == "learn:googlestock":
                embedding_dataloaders = (dataloaders_dict[idx]["full"],)  # NOTE: Could alternatively use "test" subset.
                max_seq_len = total_timesteps
            else:
                embedding_dataloaders = \
                    (dataloaders_dict[idx]["train"], dataloaders_dict[idx]["val"], dataloaders_dict[idx]["test"])
                max_seq_len = max_timesteps_list[idx]
            embeddings_filepath = os.path.join(os.path.abspath(embeddings_dir), embeddings_name_list[idx])
            embeddings = s2s_utils.get_embeddings(
                seq2seq=s2s, 
                dataloaders=embedding_dataloaders,
                padding_value=exp_settings["pad_val"],
                max_seq_len=max_seq_len
            )
            np.save(embeddings_filepath, embeddings)
            print(f"Generated and saved embeddings of shape: {embeddings.shape}. File: {embeddings_filepath}.")
    
    # Autoencoder application experiments.
    elif "apply:" in run_experiment:
        
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

        if run_experiment == "apply:amsterdam:hns_competition_data":

            for hider_name in exp_settings["hiders_list"]:
                
                # Prepare the data.
                filepath_proc_cached = os.path.join(exp_settings["proc_cached_dir"], f"{hider_name}.npz")
                if not exp_settings["load_from_proc_cached"]:
                    generated_data, seq_lens = prepare_hns_gen_data(hider_name=hider_name, exp_settings=exp_settings)
                    np.savez(filepath_proc_cached, x=generated_data, x_len=seq_lens)
                else:
                    with np.load(filepath_proc_cached) as d_full:
                        generated_data = d_full["x"]
                        seq_lens = d_full["x_len"]
                
                dataset, dataloader = get_inference_dataloader(
                    x=generated_data, 
                    x_len=seq_lens, 
                    device=selected_device, 
                    exp_settings=exp_settings)
                
                # Get autoencoder loss.
                autoencoder_loss = iterate_eval_set(
                    seq2seq=s2s, 
                    dataloader=dataloader,
                    padding_value=exp_settings["pad_val"],
                    max_seq_len=exp_settings["max_timesteps"]
                )

                # Save embeddings.
                embeddings_filepath = os.path.join(
                    os.path.abspath(embeddings_dir), 
                    exp_settings["embeddings_subdir"],
                    exp_settings["embeddings_name"].replace("<uname>", hider_name)
                )
                embeddings = s2s_utils.get_embeddings(
                    seq2seq=s2s, 
                    dataloaders=(dataloader,),
                    padding_value=exp_settings["pad_val"],
                    max_seq_len=exp_settings["max_timesteps"]
                )
                np.save(embeddings_filepath, embeddings)
                n_nan = np.isnan(embeddings).astype(int).sum()
                assert n_nan == 0

                # Print info.
                print(f"H&S submission '{hider_name}':")
                print(f"AE Loss = {autoencoder_loss:.3f}")
                print(f"Generated and saved embeddings of shape: {embeddings.shape}. File: {embeddings_filepath}.")
                print("=" * 120)
        
        elif run_experiment == "apply:amsterdam:combined_downsampled_subset":
            
            for model_name in exp_settings["models_list"]:

                filepath_gen_data = os.path.join(
                    exp_settings["gen_data_path"],
                    exp_settings["generated_data_name"].replace("<model_name>", model_name),
                )
                generated_data = np.load(filepath_gen_data)
                seq_lens = np.ones((generated_data.shape[0],), dtype=int) * generated_data.shape[1]
                # print(gen_data)
                # print(seq_lens)

                dataset, dataloader = get_inference_dataloader(
                    x=generated_data, 
                    x_len=seq_lens, 
                    device=selected_device, 
                    exp_settings=exp_settings)
                
                # Get autoencoder loss.
                autoencoder_loss = iterate_eval_set(
                    seq2seq=s2s, 
                    dataloader=dataloader,
                    padding_value=exp_settings["pad_val"],
                    max_seq_len=exp_settings["max_timesteps"]
                )

                # Save embeddings.
                embeddings_filepath = os.path.join(
                    os.path.abspath(embeddings_dir), 
                    exp_settings["embeddings_name"].replace("<model_name>", model_name)
                )
                embeddings = s2s_utils.get_embeddings(
                    seq2seq=s2s, 
                    dataloaders=(dataloader,),
                    padding_value=exp_settings["pad_val"],
                    max_seq_len=exp_settings["max_timesteps"]
                )
                np.save(embeddings_filepath, embeddings)
                n_nan = np.isnan(embeddings).astype(int).sum()
                assert n_nan == 0

                # Print info.
                print(f"Amsterdam comb+ds; model: '{model_name}':")
                print(f"AE Loss = {autoencoder_loss:.3f}")
                print(f"Generated and saved embeddings of shape: {embeddings.shape}. File: {embeddings_filepath}.")
                print("=" * 120)


if __name__ == "__main__":
    main()
