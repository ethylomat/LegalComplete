import os
import re

import pandas as pd
import spacy
from tqdm import tqdm

from src.utils.common import read_csv
from src.utils.matcher import ReferenceMatcher

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

"""
Preprocessing of files for further processing
"""


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


def preprocess(filename):
    file_path = os.path.join(DATA_DIR, filename)
    nlp = load_spacy_model("de_core_news_sm")

    if (df := read_csv(file_path)) is None:
        print("Could not preprocess (could not read data).")
        return False

    # Matching section references using ReferenceMatcher class
    reference_matcher = ReferenceMatcher()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    nlp.add_pipe(reference_matcher, before="tagger")

    sentence_references = []

    print("Finding section references ...")
    for item in tqdm(df["text"]):

        # Replacing newlines and multiple spaces
        text = re.sub(r"[ ]+", " ", item.replace("\n", " "))
        doc = nlp(text)

        for sentence in doc.sents:
            section_references = [
                ent for ent in sentence.ents if ent.label_ == "SECTION_REFERENCE"
            ]

            # Select sentences with only one reference
            if len(section_references) == 1:
                section_reference = section_references[0]
                sentence_references.append(
                    [
                        sentence.text,
                        str(section_reference),
                    ]
                )

    section_reference_df = pd.DataFrame(sentence_references)
    print(section_reference_df)


if __name__ == "__main__":
    preprocess()
