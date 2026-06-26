# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

def load_csvs(folder_1, folder_2):
    # I want to make one big .csv containing all shutdown events, with an added ID-column 
    # And perhaps timestamp column? 1-800 s
    # And also a target column

    # Get list of all CSV files in both folders
    csv_files = list(folder_1.glob("*.csv")) + list(folder_2.glob("*.csv")) # glob finds files matching specific patterns 

    all_dfs = []

    for csv_file in csv_files:
        # Example filename: Shutdown_0_36774.csv
        name = csv_file.stem  # removes .csv
        _, target, ID = name.split("_")

        target = int(target)
        ID = int(ID)

        # Read CSV
        df = pd.read_csv(csv_file)

        # Add required columns
        df["ID"] = ID
        df["timestamp"] = range(1, len(df) + 1)
        df["target"] = target

        all_dfs.append(df)

    return all_dfs

def combine_csvs(all_dfs):
    return pd.concat(all_dfs, ignore_index=True)

def clean_data(all_dfs):
    filled_dfs = []

    for df in all_dfs:
        df_filled = df.copy() # No change to the original df
        df_filled = df_filled.ffill() # forward-fill
        df_filled = df_filled.bfill() # handle NaNs at the beginning of an event 
        
        filled_dfs.append(df_filled)

    return filled_dfs

def load_blind_dataset(folder_path):
    blind_dfs = []
    for csv_file in folder_path.glob("*.csv"):
        name = csv_file.stem
        ID = name.split("_")[1]

        # Read CSV
        df = pd.read_csv(csv_file)

        # Add required columns
        df["ID"] = ID
        df["timestamp"] = range(1, len(df) + 1)

        blind_dfs.append(df)
    return blind_dfs

# LSTM class (PyTorch)
# Wrapper so that PyTorch can handle the data. --> Tensors 
class ReactorDataset(Dataset):
    def __init__(self, X, y):
        """
        X: list or array of shape (num_events, 800, 10)
        y: (num_events,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]