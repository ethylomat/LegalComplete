import os
import re

import pandas as pd
import spacy
from spacy.kb import KnowledgeBase
from spacy.vocab import Vocab
from tqdm import tqdm

from src.utils.preprocessing import read_csv
from src.utils.retrieve import get_dataset_info

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

"""
Preprocessing list of stopwords
"""


def read_stopwords_csv():
    """
    Reads in list of stopwords from csv file
    Returns:
    - df: Dataframe with stopwords in different columns
    """
    dataset_info = get_dataset_info("sw_de_rs")
    df = read_csv(os.path.join(DATA_DIR, dataset_info["extracted"]))
    return df


def stopwords_from_df(df) -> list:
    """
    Merges dataframe column values of stopwords
    Arguments:
    - df: Pandas Dataframe with stopwords in multiple columns
    Returns:
    - stopwords: Python-Array of stopwords
    """
    array = []
    for column in df.columns:
        for value in df[column].values:
            if type(value) == str:
                array.append(value)
    return array


def custom_stopwords() -> list:
    """
    Adding custom stopwords.
    Returns:
    - custom_stopwords: List containing custom stopwords.
    """
    custom_stopwords = []

    custom_stopwords += ["- %d -" % i for i in range(1, 150)]
    return custom_stopwords


def stopwords():
    """
    Main function for generation of stopword list
    Returns:
    - stopwords: List of stopwords (containing stopwords from csv and custom stopwords
    """
    df = read_stopwords_csv()
    stopwords = stopwords_from_df(df)
    stopwords += custom_stopwords()
    stopwords = list(set(stopwords))
    return stopwords


stopwords = stopwords()

if __name__ == "__main__":
    print(stopwords)
