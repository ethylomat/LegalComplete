import math
import os
import re
import timeit
from functools import wraps

import pandas as pd
import spacy

from src.utils.retrieve import download_all_datasets, get_dataset_info

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

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
    path = os.path.join(DATA_DIR, path)

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
    assert FileNotFoundError


def read_dataset(key: str):
    dataset_info = get_dataset_info(key)
    return read_csv(os.path.join(DATA_DIR, dataset_info["extracted"]))


def clean_n_gram(n_gram: list) -> list:
    """
    Cleaning n-gram (also padding missing words)
    """
    cleaned = [re.sub(r"(\d+|\W+)", "<>", str(s)) for s in n_gram]
    cleaned = [x.replace("<>", "") if x != "<>" else x for x in cleaned if x != "<>"]
    return ["<>"] * (4 - len(cleaned)) + cleaned


def load_spacy_model(model_key: str = "de_core_news_sm"):
    """
    Function for loading a spacy model. If not downloaded it downloads the model.
    Arguments:
    - model_key: spaCy model key
    Returns:
    - nlp: Spacy instane
    """
    try:
        print("Loading spaCy model ...")
        nlp = spacy.load(model_key)
    except OSError:
        print("Downloading spaCy model ...")
        spacy.cli.download(model_key)
        nlp = spacy.load(model_key)
    return nlp


def split_dataframe(df, fracs=[0.8, 0.1, 0.1]):
    """
    Function for splitting dataframes into multiple sets.
    Arguments:
    - df: Dataframe
    Returns:
    - dfs: List of Dataframes
    """
    if frac_sum := sum(fracs) > 1:
        fracs = [frac / frac_sum for frac in fracs]

    dfs = []
    for i, frac in enumerate(fracs):
        denominator = 1 - sum(fracs[:i])
        partition = df.sample(frac=round(frac / denominator, 5), random_state=0)
        df = df.drop(partition.index)
        dfs.append(partition)
    return dfs


def timer(method):
    """
    Method decorator to measure the time for method execution.

    Example useage:

    @timer
    def hello_world():
        pass
    """

    @wraps(method)
    def timing(self, *method_args, **method_kwargs):
        start_time = timeit.default_timer()
        output = method(self, *method_args, **method_kwargs)
        elapsed = timeit.default_timer() - start_time
        print(
            '-- Method "{name}" took {time} seconds to complete.'.format(
                name=method.__name__, time=elapsed
            )
        )
        return output

    return timing
