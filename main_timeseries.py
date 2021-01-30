"""Time series data generation.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
import os

import numpy as np

from generative_models.timegan import timegan
from data.amsterdam import AmsterdamLoader, preprocess_data, padding_mask_to_seq_lens

# ----------------------------------------------------------------------------------------------------------------------
# Set experiment settings here:

use_data = "amsterdam"
use_model = "timegan"

generated_data_dir = "./data/ts_generated/"

amsterdam_data_settings = {
    "train_frac": 0.4,
    "val_frac": 0.2,
    "n_features": 70,
    "include_time": False,
    "max_timesteps": 100,
    "pad_val": -999.,
    "data_split_seed": 12345,
    "data_loading_force_refresh": True,
    # --------------------
    "data_path": "data/amsterdam/combined_downsampled_longitudinal_data.csv",
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
    "generated_data_filename": "<data>_timegan.npy"  # NOTE: <data> will be replaced with `use_data` value.
}

# ----------------------------------------------------------------------------------------------------------------------
# Utilities.

def prepare_amsterdam(amsterdam_loader, settings):
    raw_data, padding_mask, (train_idx, val_idx, test_idx) = \
        amsterdam_loader.load_reshape_split_data(force_refresh=settings["data_loading_force_refresh"])
    processed_data, imputed_processed_data = preprocess_data(
        raw_data, 
        padding_mask, 
        padding_fill=settings["pad_val"],
        time_feature_included=settings["include_time"],
    )
    seq_lens = padding_mask_to_seq_lens(padding_mask)
    return imputed_processed_data, seq_lens

# ----------------------------------------------------------------------------------------------------------------------

def main():
    
    if use_data == "amsterdam":
        active_data_settings = amsterdam_data_settings
        amsterdam_loader = AmsterdamLoader(
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
        original_data, seq_lens = prepare_amsterdam(amsterdam_loader=amsterdam_loader, settings=active_data_settings)

    if use_model == "timegan":
        active_experiment_settings = timegan_experiment_settings
        generated_data = timegan(ori_data=original_data, parameters=active_experiment_settings["model_params"])
    
    generated_data_filepath = os.path.join(
        generated_data_dir, 
        active_experiment_settings["generated_data_filename"].replace("<data>", use_data))
    np.save(generated_data_filepath, generated_data)
    print(f"Generative model: {use_model}, data: {use_data}\n" 
        f"Generated and saved timeseries data of shape: {generated_data.shape}. File: {generated_data_filepath}.")


if __name__ == "__main__":
    main()
