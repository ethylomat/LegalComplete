import os

import pandas as pd
from retrieve import download_all_datasets

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

"""
Preprocessing of files for further processing
"""


def read_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    print("Path does not exist. Downloading all available datasets ...")
    download_all_datasets()
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    print("Path does not exist after downloading.")
    return None


def preprocess():
    filename = "2020_06_23_CE-BVerwG_DE_Datensatz.csv"
    path = os.path.join(DATA_DIR, filename)

    df = read_csv(path)
    if df is None:
        print("Could not preprocess (could not read data).")
        return False

    print(df.iloc[0]["text"])


if __name__ == "__main__":
    preprocess()
