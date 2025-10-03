import json
import os
import pandas as pd


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_and_combine_data(directory: str) -> pd.DataFrame:
    """
    Reads all CSV files from the given directory and appends them into a single DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame containing all CSV data.
    """
    all_dfs = []

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_csv(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if no CSVs found
