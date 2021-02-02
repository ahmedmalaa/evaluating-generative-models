"""Amsterdam UMCdb data preprocessing: scripts.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
from .data_preprocess import combine_csvs, downsample_csv_by_admissionids


# Script settings:
run_script = "combine_downsample"
filepaths = {
    "source": {
        "train_data_filepath": "./train_longitudinal_data.csv",
        "test_data_filepath": "./test_longitudinal_data.csv"
    },
    "output": {
        "out_combined_filepath": "./combined_longitudinal_data.csv",
        "out_combined_downsampled_filepath": "./combined_downsampled_longitudinal_data.csv"
    }
}
downsample_n_ids = 1000
downsample_seed = 12345


def main():
    
    if run_script == "combine_downsample":
        # Note: requires between 64 and 128 GB of memory.
        combine_csvs(
            path_train=filepaths["source"]["train_data_filepath"], 
            path_test=filepaths["source"]["test_data_filepath"],
            path_combined=filepaths["output"]["out_combined_filepath"]
        )
        downsample_csv_by_admissionids(
            path=filepaths["output"]["out_combined_filepath"],
            path_downsampled=filepaths["output"]["out_combined_downsampled_filepath"],
            downsample_n_ids=downsample_n_ids,
            seed=downsample_seed
        )


if __name__ == "__main__":
    main()
