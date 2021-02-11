"""Time series data generation.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
import os
import copy

import numpy as np

from data import amsterdam
from utils import prepare_amsterdam
from generative_models.timegan import timegan
from generative_models.rgan import rgan

# ----------------------------------------------------------------------------------------------------------------------
# Set experiment settings here:

# Options:
# use_data:
#   - "amsterdam:combined_downsampled_subset"
#   - "snp500"
# use_model:
#   - "timegan"
#   - "rgan"
#   - "rgan_dp"
use_data = "amsterdam:combined_downsampled_subset"
use_model = "rgan"

generated_data_dir = "./data/ts_generated/"

_use_amsterdam_comb_version = "1000"  # Options: ("1000", "5000")
_use_amsterdam_seq_len = 100  # Options: (10, 100, 100)
amsterdam_data_settings = {
    "train_frac": 0.4,
    "val_frac": 0.2,
    "n_features": 70,
    "include_time": False,
    "max_timesteps": _use_amsterdam_seq_len
    "pad_val": 0.,
    "data_split_seed": 12345,
    "data_loading_force_refresh": True,
    # --------------------
    "data_path": f"data/amsterdam/combined_downsampled{_use_amsterdam_comb_version}_longitudinal_data.csv",
    "embeddings_name": \
        f"amsterdam_embeddings_comb{_use_amsterdam_comb_version}_{_use_amsterdam_seq_len}"
}

timegan_experiment_settings = {
    "model_params": {
        "module": "gru",
        "hidden_dim": 10,
        "num_layer": 3,
        "iterations": 1000,
        "batch_size": 128,
        "print_every_n_iters": 100,
    },
    "generated_data_filename": "<embeddings_name>_timegan.npy"
}

rgan_experiment_settings = {
    "model_params": {
        "hidden_units_g": 100,
        "hidden_units_d": 100,
        "latent_dim": 10,
        "l2norm_bound": 1e-05,
        "learning_rate": 0.1,
        "batch_size": 28,
        "num_epochs": 100,
        "D_rounds": 1,
        "G_rounds": 3,
        # DP Settings:
        "dp": False,
        "dp_sigma": None,
    },
    "generated_data_filename": "<embeddings_name>_rgan.npy"
}

rgan_dp_experiment_settings = copy.deepcopy(rgan_experiment_settings)
rgan_dp_experiment_settings["model_params"]["dp"] = True
rgan_dp_experiment_settings["model_params"]["dp_sigma"] = 0.1  # 1e-05
rgan_dp_experiment_settings["generated_data_filename"] = "<embeddings_name>_rgan_dp.npy"


# ----------------------------------------------------------------------------------------------------------------------
# Utilities.

# Empty.

# ----------------------------------------------------------------------------------------------------------------------

def main():
    
    if use_data == "amsterdam:combined_downsampled_subset":
        active_data_settings = amsterdam_data_settings
        amsterdam_loader = amsterdam.AmsterdamLoader(
            data_path=os.path.abspath(active_data_settings["data_path"]),
            max_seq_len=active_data_settings["max_timesteps"],
            seed=active_data_settings["data_split_seed"],
            train_rate=active_data_settings["train_frac"],
            val_rate=active_data_settings["val_frac"],
            include_time=active_data_settings["include_time"],
            debug_data=False,
            pad_before=False,
            padding_fill=active_data_settings["pad_val"],
        )
        if use_model == "timegan":
            # Timegan doesn't take variable-length sequences, use padding value of 0.
            amsterdam_loader.padding_fill = 0.
        original_data, seq_lens, _ = prepare_amsterdam(amsterdam_loader=amsterdam_loader, settings=active_data_settings)
    
    else:
        raise ValueError(f"Unknown data source selected: '{use_data}'.")

    if use_model == "timegan":
        active_experiment_settings = timegan_experiment_settings
        generated_data = timegan(ori_data=original_data, parameters=active_experiment_settings["model_params"])
    
    elif use_model in ("rgan", "rgan_dp"):
        active_experiment_settings = rgan_experiment_settings if use_model == "rgan" else  rgan_dp_experiment_settings
        active_experiment_settings["model_params"]["data"] = use_data
        active_experiment_settings["model_params"]["identifier"] = use_model
        active_experiment_settings["model_params"]["custom_experiment"] = True
        # ^ Keep "custom_experiment" True, needed for the rgan() script.
        active_experiment_settings["model_params"]["num_samples"] = None  # Auto-set later.
        active_experiment_settings["model_params"]["seq_length"] = active_data_settings["max_timesteps"]
        active_experiment_settings["model_params"]["num_signals"] = active_data_settings["n_features"]
        active_experiment_settings["model_params"]["num_generated_features"] = active_data_settings["n_features"]
        generated_data = rgan(ori_data=original_data, parameters=active_experiment_settings["model_params"])
        print(f"{'RGAN' if use_model == 'rgan' else 'RGAN-DP'} Generated Data:")
        print("shape:", generated_data.shape)
        print(generated_data)
    
    else:
        raise ValueError(f"Unknown model selected: '{use_model}'.")
    
    generated_data_filepath = os.path.join(
        generated_data_dir, 
        active_experiment_settings["generated_data_filename"].replace(
            "<embeddings_name>", 
            active_data_settings["embeddings_name"]
        )
    )
    np.save(generated_data_filepath, generated_data)
    print(f"Generative model: {use_model}, data: {use_data}\n" 
        f"Generated and saved timeseries data of shape: {generated_data.shape}. File: {generated_data_filepath}.")


if __name__ == "__main__":
    main()
