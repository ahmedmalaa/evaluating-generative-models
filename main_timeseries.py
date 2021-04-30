"""Time series data generation.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
import os
import copy
import pprint

import numpy as np

from data import amsterdam
from data import snp500
from utils import prepare_amsterdam


# ----------------------------------------------------------------------------------------------------------------------
# Set experiment settings here:

# Visible GPU(s):
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # "1" or "0,2" etc.

# Options:
# use_data:
#   - "amsterdam:combds:N:T"
#   - "snp500:T"
# NOTE: 
#   See main_timeseries_embedding.py for N and T.
# use_model:
#   - "timegan"
#   - "rgan"
#   - "rgan-dp"
#   - "add-noise"
use_data = "snp500:125"
use_model = "rgan-dp"

generated_data_dir = "./data/ts_generated/"

# Import after choosing model to allow for different environments:
if use_model in ("rgan", "rgan-dp"):
    from generative_models.rgan import rgan
if use_model == "timegan":
    from generative_models.timegan import timegan
if use_model == "add-noise":
    from generative_models.add_noise import add_noise


# ----------------------------------------------------------------------------------------------------------------------
# Hyperparameters and other settings:

# NOTE: Automatically extracts N and T from experiment name:
use_data_full_name = use_data
if "amsterdam:combds" in use_data:
    _amsterdam_combds_N = use_data.split(":")[-2]
    _amsterdam_combds_T = int(use_data.split(":")[-1])
    use_data_full_name = copy.copy(use_data)
    use_data = "amsterdam:combds"
else:
    _amsterdam_combds_N = "NOT_SET"
    _amsterdam_combds_T = "NOT_SET"
amsterdam_experiment_settings = {
    "data": {
        "train_frac": 0.4,
        "val_frac": 0.2,
        "n_features": 70,
        "include_time": False,
        "max_timesteps": _amsterdam_combds_T,
        "pad_val": 0.5,
        "data_split_seed": 12345,
        "data_loading_force_refresh": True,
        # --------------------
        "data_path": f"data/amsterdam/combined_downsampled{_amsterdam_combds_N}_longitudinal_data.csv",
        "original_copy_filename": f"amsterdam-combds-{_amsterdam_combds_N}-{_amsterdam_combds_T}_ORIGINAL.npy",
        "generated_name": \
            f"amsterdam-combds-{_amsterdam_combds_N}-{_amsterdam_combds_T}_generated"
    },
    "models": {
        "timegan": {
            "model_params": {
                "module": "gru",
                "hidden_dim": 10,
                "num_layer": 3,
                "iterations": 2_000,
                "batch_size": 1024,
                "print_every_n_iters": 100,
            },
            "generated_data_filename": "<generated_name>_timegan.npy"
        },
        "rgan": {
            "model_params": {
                "hidden_units_g": 100,
                "hidden_units_d": 100,
                "latent_dim": 10,
                "l2norm_bound": 1e-05,
                "learning_rate": 0.1,
                "batch_size": 256,
                "num_epochs": 1_000,
                "D_rounds": 1,
                "G_rounds": 3,
                # DP Settings:
                "dp": False,
                "dp_sigma": None,
            },
            "generated_data_filename_best": "<generated_name>_rgan_best.npy",
            "generated_data_filename_last": "<generated_name>_rgan_last.npy",
        },
        "rgan-dp": {
                "model_params": {
                "hidden_units_g": 100,
                "hidden_units_d": 100,
                "latent_dim": 10,
                "l2norm_bound": 1e-05,
                "learning_rate": 0.1,
                "batch_size": 128,
                "num_epochs": 500,
                "D_rounds": 3,
                "G_rounds": 1,
                # DP Settings:
                "dp": True,
                "dp_sigma": [1e-01, 1e-03, 1e-05]  # Options: one or more (as list) from [1e-01, 1e-03, 1e-05]
            },
            "generated_data_filename_best": "<generated_name>_rgan-dp-<sigma>_best.npy",
            "generated_data_filename_last": "<generated_name>_rgan-dp-<sigma>_last.npy",
        },
        "add-noise": {
            "model_params": {
                "sigma": [0.1, 0.001, 0.00001]
            },
            "generated_data_filename": "<generated_name>_add-noise-<sigma>.npy",
        }
    }
}

if "snp500" in use_data:
    _snp500_T = int(use_data.split(":")[-1])
    use_data = "snp500"
else:
    _snp500_T = "NOT_SET"
snp500_experiment_settings = {
    "data": {
        "train_frac": 0.4,
        "val_frac": 0.2,
        "n_features": 5,
        "include_time": False,
        "max_timesteps": 1259,
        "pad_val": 0.,
        "data_split_seed": 12345,
        "data_loading_force_refresh": True,
        # --------------------
        "data_path": "./data/snp500/all_stocks_5yr.csv",
        "npz_cache_filepath": f"./data/snp500/snp500-{_snp500_T}.npz",
        "original_copy_filename": f"snp500-{_snp500_T}_ORIGINAL.npy",
        "generated_name": f"snp500-{_snp500_T}_generated"
    },
    "models": {
        "timegan": {
            "model_params": {
                "module": "gru",
                "hidden_dim": 10,
                "num_layer": 3,
                "iterations": 2_000,
                "batch_size": 1024,
                "print_every_n_iters": 100,
            },
            "generated_data_filename": "<generated_name>_timegan.npy"
        },
        "rgan": {
            "model_params": {
                "hidden_units_g": 100,
                "hidden_units_d": 100,
                "latent_dim": 10,
                "l2norm_bound": 1e-05,
                "learning_rate": 0.1,
                "batch_size": 64,
                "num_epochs": 500,
                "D_rounds": 1,
                "G_rounds": 6,
                # DP Settings:
                "dp": False,
                "dp_sigma": None,
            },
            "generated_data_filename_best": "<generated_name>_rgan_best.npy",
            "generated_data_filename_last": "<generated_name>_rgan_last.npy",
        },
        "rgan-dp": {
                "model_params": {
                "hidden_units_g": 100,
                "hidden_units_d": 100,
                "latent_dim": 10,
                "l2norm_bound": 1e-05,
                "learning_rate": 0.1,
                "batch_size": 16,
                "num_epochs": 100,
                "D_rounds": 4,
                "G_rounds": 1,
                # DP Settings:
                "dp": True,
                "dp_sigma": [1e-01, 1e-05]  # Options: one or more (as list) from [1e-01, 1e-03, 1e-05]
            },
            "generated_data_filename_best": "<generated_name>_rgan-dp-<sigma>_best.npy",
            "generated_data_filename_last": "<generated_name>_rgan-dp-<sigma>_last.npy",
        },
        "add-noise": {
            "model_params": {
                "sigma": [0.1, 0.001, 0.00001]
            },
            "generated_data_filename": "<generated_name>_add-noise-<sigma>.npy",
        }
    }
}


# ----------------------------------------------------------------------------------------------------------------------
# Utilities.

def print_exp_info(data_used, model_used, data_settings, exp_settings):
    print("=" * 80)
    print(f"\nExperiment: data='{data_used}' model='{model_used}'")
    print("\nData settings:\n")
    pprint.pprint(data_settings, indent=4)
    print("\nModel settings:\n")
    pprint.pprint(exp_settings, indent=4)
    print()
    print("=" * 80)


# ----------------------------------------------------------------------------------------------------------------------

def main():

    # Collect settings:
    if use_data == "amsterdam:combds":
        active_experiment_settings = amsterdam_experiment_settings
        active_data_settings = amsterdam_experiment_settings["data"]
    elif use_data == "snp500":
        active_experiment_settings = snp500_experiment_settings
        active_data_settings = snp500_experiment_settings["data"]
    else:
        raise ValueError(f"Unknown data source selected: '{use_data}'.")
    
    if use_model == "timegan":
        active_model_settings = active_experiment_settings["models"]["timegan"]
    elif use_model == "rgan":
        active_model_settings = active_experiment_settings["models"]["rgan"]
    elif use_model == "rgan-dp":
        active_model_settings = active_experiment_settings["models"]["rgan-dp"]
    elif use_model == "add-noise":
        active_model_settings = active_experiment_settings["models"]["add-noise"]
    else:
        raise ValueError(f"Unknown model selected: '{use_model}'.")

    print_exp_info(use_data_full_name, use_model, active_data_settings, active_model_settings)
    
    # Prepare data:
    if use_data == "amsterdam:combds":
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
        original_data, seq_lens, _ = prepare_amsterdam(amsterdam_loader=amsterdam_loader, settings=active_data_settings)
    
    elif use_data == "snp500":
        original_data, seq_lens, _ = snp500.load_snp_data(
            data_path=active_data_settings["data_path"], 
            npz_cache_filepath=active_data_settings["npz_cache_filepath"], 
            padding_value=active_data_settings["pad_val"], 
            normalize=True, 
            include_time=active_data_settings["include_time"], 
            force_refresh=active_data_settings["data_loading_force_refresh"], 
        )
    
    original_copy_path = os.path.join(generated_data_dir, active_data_settings["original_copy_filename"])
    if not os.path.exists(original_copy_path):
        np.save(original_copy_path, original_data)
        print(f"Original data copy saved for: {use_data_full_name}\n" 
            f"Data shape: {original_data.shape}. File: {original_copy_path}.")

    # Run model:
    if use_model == "timegan":
        generated_data = timegan(ori_data=original_data, parameters=active_model_settings["model_params"])

        generated_data_filepath = os.path.join(
            generated_data_dir, 
            active_model_settings["generated_data_filename"].replace(
                "<generated_name>", 
                active_data_settings["generated_name"]
            )
        )
        np.save(generated_data_filepath, generated_data)
        print(f"Generative model: {use_model}, data: {use_data}\n" 
            f"Generated and saved timeseries data of shape: {generated_data.shape}. File: {generated_data_filepath}.")
    
    elif use_model in ("rgan", "rgan-dp"):
        
        active_model_settings["model_params"]["data"] = use_data
        active_model_settings["model_params"]["identifier"] = use_model
        
        active_model_settings["model_params"]["custom_experiment"] = True
        # ^ Keep "custom_experiment" True, needed for the rgan() script.
        
        active_model_settings["model_params"]["num_samples"] = None  # Auto-set later.
        active_model_settings["model_params"]["seq_length"] = active_data_settings["max_timesteps"]
        active_model_settings["model_params"]["num_signals"] = active_data_settings["n_features"]
        active_model_settings["model_params"]["num_generated_features"] = active_data_settings["n_features"]
        
        if isinstance(active_model_settings["model_params"]["dp_sigma"], (list, tuple)):
            sigmas = active_model_settings["model_params"]["dp_sigma"]
            assert use_model == "rgan-dp"
        else:
            sigmas = [active_model_settings["model_params"]["dp_sigma"]]
        
        for sigma in sigmas:
            
            active_model_settings["model_params"]["dp_sigma"] = sigma
            if use_model == 'rgan':
                print("Running RGAN...")
            else:
                print(f"Running RGAN-DP (sigma={sigma})...")
            
            generated_data_best, generated_data_last = rgan(
                ori_data=original_data, 
                parameters=active_model_settings["model_params"])
            if use_model == 'rgan':
                print("RGAN Generated Data:")
            else:
                print(f"RGAN-DP (sigma={sigma}) Generated Data:")
            print("shape:", generated_data_best.shape)
            print(generated_data_best)

            generated_data_filepath_best = os.path.join(
                generated_data_dir, 
                active_model_settings["generated_data_filename_best"].replace(
                    "<generated_name>", 
                    active_data_settings["generated_name"]
                )
            )
            generated_data_filepath_last = os.path.join(
                generated_data_dir, 
                active_model_settings["generated_data_filename_last"].replace(
                    "<generated_name>", 
                    active_data_settings["generated_name"]
                )
            )
            if use_model == 'rgan-dp':
                generated_data_filepath_best = generated_data_filepath_best.replace("<sigma>", f"s{sigma:.0e}")
                generated_data_filepath_last = generated_data_filepath_last.replace("<sigma>", f"s{sigma:.0e}")
            np.save(generated_data_filepath_best, generated_data_best)
            np.save(generated_data_filepath_last, generated_data_last)
            print(f"Generative model: {use_model}, data: {use_data}\n" 
                f"Generated and saved timeseries data of shape: {generated_data_best.shape}. "
                f"File: {generated_data_filepath_best}.")
            print(f"Generative model: {use_model}, data: {use_data}\n" 
                f"Generated and saved timeseries data of shape: {generated_data_last.shape}. "
                f"File: {generated_data_filepath_last}.")
    
    elif use_model == "add-noise":
        
        if isinstance(active_model_settings["model_params"]["sigma"], (list, tuple)):
            sigmas = active_model_settings["model_params"]["sigma"]
        else:
            sigmas = [active_model_settings["model_params"]["sigma"]]
        
        for sigma in sigmas:
            generated_data = add_noise(ori_data=original_data, noise_size=sigma)

            generated_data_filepath = os.path.join(
                generated_data_dir, 
                active_model_settings["generated_data_filename"].replace(
                    "<generated_name>", 
                    active_data_settings["generated_name"]
                ).replace("<sigma>", f"{sigma:.0e}")
            )

            np.save(generated_data_filepath, generated_data)
            print(f"Generative model: {use_model}, data: {use_data}\n" 
                f"Generated and saved timeseries data of shape: {generated_data.shape}. "
                f"File: {generated_data_filepath}.")
    
    # Unset visible GPU env. variable.
    if "CUDA_VISIBLE_DEVICES" in os.environ: 
        del os.environ["CUDA_VISIBLE_DEVICES"]


if __name__ == "__main__":
    main()
