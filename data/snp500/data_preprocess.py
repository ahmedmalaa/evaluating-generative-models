import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_snp_data(
    data_path, 
    npz_cache_filepath=None,
    padding_value=-999., 
    normalize=True, 
    include_time=False,
    force_refresh=True,
):
    if not force_refresh:
        assert npz_cache_filepath is not None, "If not re-processing, need `npz_cache_filepath` to load from."

    if force_refresh:
        # Load.
        df = pd.read_csv(os.path.abspath(data_path), parse_dates=["date"])
        df["volume"] = df["volume"].astype(float)
        
        # Get stocks list, and possible time lengths present.
        stocks = sorted(list(df["Name"].unique()))
        stock_count = df.groupby("Name").count()
        time_lengths = np.unique(stock_count.to_numpy())

        # Preprocess.
        seq_lens = np.zeros((len(stocks),), dtype=int)
        processed_data = np.full((len(stocks), max(time_lengths), 6 if include_time else 5), padding_value)
        stock_idx_to_name = dict()

        for idx, stock in enumerate(stocks):
            stock_idx_to_name[idx] = stock
            
            df_ = df[df["Name"] == stock].reset_index(drop=True).drop(columns=["Name"])

            if df_.isnull().sum().sum() > 0:
                df_ = df_.fillna(method='ffill')
                df_ = df_.fillna(method='bfill')
                assert df_.isnull().sum().sum() == 0

            array = df_[[*tuple(df_.columns[1:])]].to_numpy()  # Values
            time = df_[["date"]].to_numpy()

            # Preprocess time:
            time = (time - np.min(time)).astype('timedelta64[D]').astype(float)  # Time delta in days.
            
            # Preprocess values:
            if normalize:
                scaler = MinMaxScaler()
                array = scaler.fit_transform(array)

            # Combine:
            if include_time:
                array = np.concatenate([time, array], axis=1)

            seq_lens[idx] = array.shape[0]
            processed_data[idx, :array.shape[0], :] = array
        
        # Save.
        if npz_cache_filepath is not None:
            np.savez(
                os.path.abspath(npz_cache_filepath), 
                processed_data=processed_data, 
                seq_lens=seq_lens, 
                stock_idx_to_name=stock_idx_to_name
            )
    
    else:
        # Load from cached file.
        with np.load(os.path.abspath(npz_cache_filepath), allow_pickle=True) as d:
            processed_data = d["processed_data"]
            seq_lens = d["seq_lens"]
            stock_idx_to_name = d["stock_idx_to_name"]

    return processed_data, seq_lens, stock_idx_to_name


def split_snp_data(data, seq_lens, frac_train, frac_val, seed):
    assert frac_train > 0. and frac_train < 1.
    assert frac_val >= 0. and frac_val < 1.
    
    frac_test = 1. - frac_train - frac_val
    assert frac_test + frac_val + frac_train == 1.

    frac_train_val_of_all = frac_train + frac_val
    frac_train_of_train_val = frac_train / frac_train_val_of_all

    data_, data_test, seq_lens_, seq_lens_test = \
        train_test_split(data, seq_lens, train_size=frac_train_val_of_all, shuffle=True, random_state=seed)
    data_train, data_val, seq_lens_train, seq_lens_val = \
        train_test_split(data_, seq_lens_, train_size=frac_train_of_train_val, shuffle=True, random_state=seed + 1)

    return (data_train, seq_lens_train), (data_val, seq_lens_val), (data_test, seq_lens_test)
