import os
import re

import pandas as pd
from tqdm import tqdm

from src.utils.common import load_spacy_model, read_csv
from src.utils.matcher import ReferenceMatcher

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

"""
Preprocessing of files for further processing
"""


def build_pipeline():
    """
    Function that creates the pipeline for the creation of (sentence, reference) tuples.
    Returns:
    - nlp: spaCy pipeline instance
    """
    nlp = load_spacy_model("de_core_news_sm")

    # Matching section references using ReferenceMatcher class
    reference_matcher = ReferenceMatcher()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    nlp.add_pipe(reference_matcher, before="tagger")
    return nlp


def preprocess(filename):
    """
    Preprocessing of csv files containing judgements. Expecting column “text” for the column of judgement raw texts.
    Creates tuples of sentences and section references. Surjective mapping (sentence -> section reference).
    Arguments:
    - filename: Filename of the csv file to preprocess
    Returns:
    - sentence_reference_df: Dataframe containing list of ("sentence", "reference")
    """

    file_path = os.path.join(DATA_DIR, filename)
    nlp = build_pipeline()

    if (df := read_csv(file_path)) is None:
        print("Could not preprocess (could not read data).")
        return False

    sentence_reference_pairs = []

    print("Finding section references ...")
    for item in tqdm(df["text"]):

        # Replacing newlines and multiple spaces
        text = re.sub(r"[ ]+", " ", item.replace("\n", " "))
        doc = nlp(text)

        for sentence in doc.sents:
            references = [
                ent for ent in sentence.ents if ent.label_ == "SECTION_REFERENCE"
            ]

            # Select sentences with only one reference
            if len(references) == 1:
                section_reference = references[0]
                sentence_reference_pairs.append(
                    [
                        sentence.text,
                        str(section_reference),
                    ]
                )

    sentence_reference_df = pd.DataFrame(
        sentence_reference_pairs, columns=["sentence", "reference"]
    )
    return sentence_reference_df
