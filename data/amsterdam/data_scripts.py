"""Amsterdam UMCdb data preprocessing: scripts.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
try:
    from .data_preprocess import combine_csvs, downsample_csv_by_admissionids
except ImportError:
    from data_preprocess import combine_csvs, downsample_csv_by_admissionids

# ----------------------------------------------------------------------------------------------------------------------
# Script settings, set here:
# NOTE: Script needs to be run from the directory: [repo_root]/data/amsterdam

run_script = "combine_downsample"  # Options: ("combine_downsample", "combine", "downsample")

version = "5000"  # Options: ("5000", "1000")
downsample_seed = 12345

filepaths = {
    "source": {
        "train_data_filepath": "./train_longitudinal_data.csv",
        "test_data_filepath": "./test_longitudinal_data.csv"
    },
    "output": {
        "out_combined_filepath": "./combined_longitudinal_data.csv",
        "out_combined_downsampled_filepath": \
            f"./combined_downsampled{'5000' if version == '5000' else '1000'}_longitudinal_data.csv"
    }
}

downsample_n_ids = 5000 if version == "5000" else 1000

# ----------------------------------------------------------------------------------------------------------------------


def main():
    
    if run_script in ("combine_downsample", "combine"):
        # NOTE: requires between 64 and 128 GB of memory.
        combine_csvs(
            path_train=filepaths["source"]["train_data_filepath"], 
            path_test=filepaths["source"]["test_data_filepath"],
            path_combined=filepaths["output"]["out_combined_filepath"]
        )
    if run_script in ("combine_downsample", "downsample"):
        downsample_csv_by_admissionids(
            path=filepaths["output"]["out_combined_filepath"],
            path_downsampled=filepaths["output"]["out_combined_downsampled_filepath"],
            downsample_n_ids=downsample_n_ids,
            seed=downsample_seed
        )


if __name__ == "__main__":
    main()
