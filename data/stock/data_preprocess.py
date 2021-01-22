import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _read_stock_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"], thousands=",")
    df["Volume"] = df["Volume"].astype("float")
    return df


def load_stock_data(train_path, test_path, normalize=True, time=False):
    train_path = os.path.abspath(train_path)
    test_path = os.path.abspath(test_path)

    df_train = _read_stock_csv(train_path)
    df_test = _read_stock_csv(test_path)

    df = df_train.append(df_test, ignore_index=True)  # Combine so that can do custom train/val/test split.

    df["Date"] = (df["Date"] - df["Date"].min())  / np.timedelta64(1, "D")  # Days since start.

    data = df.to_numpy()

    if normalize:
        scaler = MinMaxScaler()
        data_no_time = data[:, 1:]
        scaler.fit(data_no_time)
        data[:, 1:] = scaler.transform(data_no_time)
    
    if not time:
        data = data[:, 1:]
    
    return data


def split_stock_data(data, frac_train, frac_val):
    
    assert frac_train > 0. and frac_train < 1.
    assert frac_val >= 0. and frac_val < 1.
    
    frac_test = 1. - frac_train - frac_val
    assert frac_test + frac_val + frac_train == 1.
    
    frac_train_of_train_val = frac_train / (frac_val + frac_train)
    
    # Note that shuffle=False.
    data_train_val, data_test = train_test_split(data, train_size=frac_train, shuffle=False)
    data_train, data_val = train_test_split(data_train_val, train_size=frac_train_of_train_val, shuffle=False)
    
    return data_train, data_val, data_test
