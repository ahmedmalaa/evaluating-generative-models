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


DEFAULT_SPLIT_ORDER = {
    "train": 1,
    "val": 2,
    "test": 3,
}


def split_stock_data(data, frac_train, frac_val, split_order=None):

    assert frac_train > 0. and frac_train < 1.
    assert frac_val >= 0. and frac_val < 1.
    
    frac_test = 1. - frac_train - frac_val
    assert frac_test + frac_val + frac_train == 1.
    
    frac_dict = dict()
    for k, v in split_order.items():
        if k == "train":
            frac_dict[v] = frac_train
        elif k == "val":
            frac_dict[v] = frac_val
        else:
            frac_dict[v] = frac_test

    #print(frac_dict)
    frac_1_2_of_all = frac_dict[1] + frac_dict[2]
    frac_1_of_1_2 = frac_dict[1] / frac_1_2_of_all
    #print("frac_1_of_1_2", frac_1_of_1_2)

    if split_order is None:
        split_order = DEFAULT_SPLIT_ORDER
    assert tuple(sorted(list(split_order.keys()))) == ("test", "train", "val")
    assert tuple(sorted(list(split_order.values()))) == (1, 2, 3)

    # Note that shuffle=False.
    data_1_2, data_3 = train_test_split(data, train_size=frac_1_2_of_all, shuffle=False)
    data_1, data_2 = train_test_split(data_1_2, train_size=frac_1_of_1_2, shuffle=False)
    
    split_content = dict()
    for k, v in split_order.items():
        if v == 1:
            split_content[k] = data_1
        elif v == 2:
            split_content[k] = data_2
        else:
            split_content[k] = data_3
    
    print("Split Google Stock data over time in fractions:\n"
        f"'train'={frac_train:.3f}, 'val'={frac_val:.3f}, 'test'={frac_test:.3f}\n"
        f"and the subsets are in the following chronological order: {split_order}")

    return split_content["train"], split_content["val"], split_content["test"]
