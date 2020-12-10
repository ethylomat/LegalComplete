import os
import re

import pandas as pd
import spacy
from utils.retrieve import download_all_datasets

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

"""
Preprocessing of files for further processing
"""


class ReferenceMatcher(object):
    """
    Class for matching section-references (§) in SpaCy.
    """

    name = "reference_matcher"
    expression = r"§ (\d+)"

    def __call__(self, doc):
        for match in re.finditer(self.expression, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end, label="SECTION_REFERENCE")
            if span is not None:
                doc.ents = list(doc.ents) + [span]
        return doc


def read_csv(path):
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
    print("Path does not exist. Downloading all available datasets ...")
    download_all_datasets()
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    print("Path does not exist after downloading.")
    return None


def preprocess():
    filename = "example_dataset.csv"
    path = os.path.join(DATA_DIR, filename)

    df = read_csv(path)

    if df is None:
        print("Could not preprocess (could not read data).")
        return False

    text = df.iloc[9]["text"][815:883]

    try:
        nlp = spacy.load("de_core_news_sm")
    except Exception:
        spacy.cli.download("de_core_news_sm")
        nlp = spacy.load("de_core_news_sm")

    sentence = text
    print(sentence)

    # Matching section references using ReferenceMatcher class
    entity_matcher = ReferenceMatcher()
    nlp.add_pipe(entity_matcher, after="ner")

    doc = nlp(sentence)

    print([(ent.text, ent.label_) for ent in doc.ents])


if __name__ == "__main__":
    preprocess()
