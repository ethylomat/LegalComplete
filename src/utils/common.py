import os
import re

import pandas as pd

from src.utils.retrieve import download_all_datasets

"""
Common used functions for data processing and handling
"""


def read_csv(path, download=True):
    """
    Function for reading csv file from DATA_DIR
    Arguments:
    - path: path of file
    Return:
    - dataframe: Panda dataframe of csv file
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    print("Path does not exist.")
    if download:
        download_all_datasets()
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    print("Path does not exist after downloading.")
    return None


def clean_n_gram(n_gram: list) -> list:
    """
    Cleaning n-gram (also padding missing words)
    """
    cleaned = [re.sub(r"(\d+|\W+)", "<>", str(s)) for s in n_gram]
    cleaned = [x.replace("<>", "") if x != "<>" else x for x in cleaned if x != "<>"]
    return ["<>"] * (4 - len(cleaned)) + cleaned
