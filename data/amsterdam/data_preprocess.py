import os
from typing import Union, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from .data_utils import data_division


def _to_3d(arr: np.ndarray, max_seq_len: int) -> np.ndarray:
    n_patients = arr.shape[0] // max_seq_len
    dim = arr.shape[1]
    return np.reshape(arr, [n_patients, max_seq_len, dim])


def _to_2d(arr: np.ndarray) -> np.ndarray:
    n_patients = arr.shape[0]
    max_seq_len = arr.shape[1]
    dim = arr.shape[2]
    return np.reshape(arr, [n_patients * max_seq_len, dim])


class AmsterdamLoader(object):
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: int,
        seed: int,
        train_rate: float,
        include_time: bool,
        debug_data: Union[int, bool] = False,
        padding_fill: float = -1.,
    ) -> None:
        """Initialise Amsterdam data loader. Here, the Amsterdam data refers to the Hide-and-Seek competition subset 
        ot the Amsterdam UMCdb dataset, specifically `train_longitudinal_data.csv` or `test_longitudinal_data.csv`.

        Args:
            data_path (str): Data CSV file path.
            max_seq_len (int): Maximum sequence length of the time series dimension - for reshaping.
            seed (int): Random seed for data split.
            train_rate (float): The fraction of the data to allocate to training set.
            include_time (bool): Whether to include time as the 0th feature in each example.
            debug_data (Union[int, bool], optional): If int, read only top debug_data-many rows, if True, 
                read only top 10000 rows, if False read whole dataset. Defaults to False.
            padding_fill (float, optional): Pad timeseries vectors shorter than max_seq_len with this value. 
                Defaults to -1.
        """
        self.data_path = os.path.abspath(data_path)
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.train_rate = train_rate
        self.include_time = include_time
        self.debug_data = debug_data
        self.padding_fill = padding_fill

    def load_reshape_split_data(self, force_reload: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load prepared data, reshape to a 3D 3D array of shape [num_examples, max_seq_len, num_features], 
        split into train, validation sets. Preprocessing of the data is done separately using `preprocess_data()`.

        Args:
            force_reload (bool): If True, will rerun this from scratch, rather than using results cached in npz file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: raw_data, padding_mask, train_idx, test_idx
        """

        npz_path = self.data_path.replace(".csv", ".npz")

        if os.path.exists(npz_path) and not force_reload:
            
            print(f"Found existing cached .npz file ({npz_path}), using cached data. Set force_reload=True to reprocess.")
            with np.load(npz_path) as data:
                raw_data = data["raw_data"]
                padding_mask = data["padding_mask"]
                train_idx = data["train_idx"]
                test_idx = data["test_idx"]

        else:

            raw_data, padding_mask = self._load_and_reshape(self.data_path)
            _, (train_idx, test_idx) = data_division(raw_data, seed=self.seed, divide_rates=[self.train_rate, 1 - self.train_rate])

            np.savez(npz_path, raw_data=raw_data, padding_mask=padding_mask, train_idx=train_idx, test_idx=test_idx)

        return raw_data, padding_mask, train_idx, test_idx

    def _load_and_reshape(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from `file_name` and reshape into a 3D array of shape [num_examples, max_seq_len, num_features].
        A padding mask of data will also be produced (same shape), having elements True where time series were padded 
        (due to being shorter than max_seq_len).

        Note:
            The 0th feature is time.

        Args:
            file_name (str): Original data CSV file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: [0] loaded and reshaped data, [1] corresponding padding.
        """
        padding_indicator = -999.0  # This value avoids clashing with any actual data.

        # Load data
        if self.debug_data is not False:
            if isinstance(self.debug_data, bool):
                nrows: Union[int, None] = 10000
            else:
                assert isinstance(self.debug_data, int), "debug_data argument must be bool or int."
                nrows = self.debug_data
        else:
            nrows = None
        ori_data = pd.read_csv(file_name, nrows=nrows)
        if ori_data.columns[0] == "Unnamed: 0":  # Remove spurious column, so that column 0 is now 'admissionid'.
            ori_data = ori_data.drop(["Unnamed: 0"], axis=1)

        # Drop time column if requested.
        if not self.include_time:
            ori_data = ori_data.drop(["time"], axis=1)

        # Parameters
        uniq_id = np.unique(ori_data["admissionid"])
        no = len(uniq_id)
        dim = len(ori_data.columns) - 1

        # Output initialization
        assert np.any(ori_data == padding_indicator) == False, f"Padding indicator value {padding_indicator} found in data"
        loaded_data = np.empty([no, self.max_seq_len, dim])  # Shape: [no, max_seq_len, dim]
        loaded_data.fill(padding_indicator)

        # For each uniq id
        for i in tqdm(range(no)):

            # Extract the time-series data with a certain admissionid
            idx = ori_data.index[ori_data["admissionid"] == uniq_id[i]]
            curr_data = ori_data.iloc[idx].to_numpy()  # Shape: [curr_no, dim + 1]

            # Assign to the preprocessed data (Excluding ID)
            curr_no = len(curr_data)
            if curr_no >= self.max_seq_len:
                loaded_data[i, :, :] = curr_data[:self.max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
            else:
                loaded_data[i, -curr_no:, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]

        padding_mask = loaded_data == padding_indicator
        loaded_data = np.where(padding_mask, self.padding_fill, loaded_data)

        return loaded_data, padding_mask

    def preprocess_data(self, data: np.ndarray, padding_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess and impute `data`.

        Note:
            The 0th feature is time, and it is preprocessed differently to the other features: not normalized to [0, 1]
            but shifted by -max_time_for_example.

        Args:
            data (np.ndarray of float): 
                Data as loaded (and reshaped to 3D). Shape [num_examples, max_seq_len, num_features].
            padding_mask (np.ndarray of bool): 
                Padding mask of data, indicating True where time series were shorter than max_seq_len and were padded. 
                Same shape as data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: [0] preprocessed data, [1] preprocessed and imputed data.
        """
        median_vals = self._get_medians(data, padding_mask)
        imputed_data = self._impute(data, padding_mask, median_vals)

        scaler_imputed = self._get_scaler(imputed_data, padding_mask)
        imputed_processed_data = self._preprocess(imputed_data, padding_mask, scaler_imputed)

        scaler_original = self._get_scaler(data, padding_mask)
        processed_data = self._preprocess(data, padding_mask, scaler_original)

        return processed_data, imputed_processed_data

    def _imputation(self, curr_data: np.ndarray, median_vals: np.ndarray, zero_fill: bool = True) -> np.ndarray:
        """Impute missing data using bfill, ffill and median imputation.

        Args:
            curr_data (np.ndarray): Data before imputation.
            median_vals (np.ndarray): Median values for each column.
            zero_fill (bool, optional): Whather to Fill with zeros the cases where median_val is nan. Defaults to True.

        Returns:
            np.ndarray: Imputed data.
        """

        curr_data = pd.DataFrame(data=curr_data)
        median_vals = pd.Series(median_vals)

        # Backward fill
        imputed_data = curr_data.bfill(axis="rows")
        # Forward fill
        imputed_data = imputed_data.ffill(axis="rows")
        # Median fill
        imputed_data = imputed_data.fillna(median_vals)

        # Zero-fill, in case the `median_vals` for a particular feature is `nan`.
        if zero_fill:
            imputed_data = imputed_data.fillna(0.0)

        if imputed_data.isnull().any().any():
            raise ValueError("NaN values remain after imputation")

        return imputed_data.to_numpy()

    def _get_medians(self, data: np.ndarray, padding_mask: np.ndarray):
        assert len(data.shape) == 3

        max_seq_len = data.shape[1]
        data = _to_2d(data)
        if padding_mask is not None:
            padding_mask = _to_2d(padding_mask)
            data_temp = np.where(padding_mask, np.nan, data)  # To avoid PADDING_INDICATOR affecting results.
        else:
            data_temp = data

        # Medians
        median_vals = np.nanmedian(data_temp, axis=0)  # Shape: [dim + 1]

        return median_vals

    def _get_scaler(self, data: np.ndarray, padding_mask: np.ndarray):
        assert len(data.shape) == 3

        max_seq_len = data.shape[1]
        data = _to_2d(data)
        if padding_mask is not None:
            padding_mask = _to_2d(padding_mask)
            data_temp = np.where(padding_mask, np.nan, data)  # To avoid PADDING_INDICATOR affecting results.
        else:
            data_temp = data

        # Scaler
        scaler = MinMaxScaler()
        scaler.fit(data_temp)  # Note that np.nan's will be left untouched.

        return scaler

    def _impute(self, data: np.ndarray, padding_mask: np.ndarray, median_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        assert len(data.shape) == 3

        data_imputed_ = np.zeros_like(data)

        for i in range(data.shape[0]):
            cur_data = data[i, :, :]
            if padding_mask is not None:
                cur_data = np.where(padding_mask[i, :, :], np.nan, cur_data)

            # Scale and impute (excluding time)
            cur_data_imputed = self._imputation(cur_data, median_vals)

            # Update
            data_imputed_[i, :, :] = cur_data_imputed

        # Set padding
        if padding_mask is not None:
            data_imputed_ = np.where(padding_mask, self.padding_fill, data_imputed_)

        return data_imputed_

    def _preprocess(self, data: np.ndarray, padding_mask: np.ndarray, scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:

        assert len(data.shape) == 3

        data_ = np.zeros_like(data)

        for i in range(data.shape[0]):
            cur_data = data[i, :, :]
            if padding_mask is not None:
                cur_data = np.where(padding_mask[i, :, :], np.nan, cur_data)

            # Preprocess time (0th element of dim. 2):
            if self.include_time:
                preprocessed_time = cur_data[:, 0] - np.nanmin(cur_data[:, 0])

            # Scale and impute (excluding time)
            cur_data = scaler.transform(cur_data)

            # Set time
            if self.include_time:
                cur_data[:, 0] = preprocessed_time

            # Update
            data_[i, :, :] = cur_data

        # Set padding
        if padding_mask is not None:
            data_ = np.where(padding_mask, self.padding_fill, data_)

        return data_
