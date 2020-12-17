import os

import spacy

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

    # TODO: More sophisticated generation of stopwords
    custom_stopwords = [
        "die",
        "des",
        "auf",
        "aus",
        "beruht",
        "der",
        "folgt",
        "im",
        "sinne",
        "in",
        "nach",
        "gegen",
        "bedeutung",
        "nicht",
        "eine",
        "gemäß",
        "den",
        "abs",
        "von",
        "ist",
        "satz",
        "zulassungsgrund",
        "anforderungen",
        "sie",
        "vgl",
    ]
    return custom_stopwords


def stopwords(nlp=None):
    """
    Main function for generation of stopword list
    Returns:
    - stopwords:
        List of stopwords (containing stopwords from csv and custom stopwords)
        If spaCy model provided, stopwords will be added to model stopwords.
    """
    stopwords_df = read_stopwords_csv()
    sws = stopwords_from_df(stopwords_df)
    sws += custom_stopwords()
    sws = list(set(sws))

    if nlp:
        model_stopwords = nlp.Defaults.stop_words
        for word in sws:
            model_stopwords.add(word)
        return model_stopwords
    return sws
