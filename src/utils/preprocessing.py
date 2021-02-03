import os
import re

import pandas as pd
from tqdm import tqdm

from src.utils.common import load_spacy_model, read_csv
from src.utils.matcher import ReferenceMatcher
from src.utils.regular_expressions import reference_pattern

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

"""
Preprocessing of files for further processing
"""


def build_pipeline(disable: list = []):
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
    nlp.disable_pipes(*disable)

    print("\nActivated pipes:")
    print(nlp.pipe_names)
    return nlp


def preprocess(df, nlp=None, label: str = ""):
    """
    Preprocessing of dataframes containing judgements. Expecting column “text” for the column of judgement raw texts.
    Creates tuples of sentences and section references. Surjective mapping (sentence -> section reference).
    Arguments:
    - df: Dataframe to preprocess
    Returns:
    - sentence_reference_df: Dataframe containing list of ("sentence", "reference")
    """

    if not nlp:
        nlp = build_pipeline()

    sentence_reference_pairs = []

    for item in tqdm(df["text"], desc=label):

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
                        sentence,
                        str(section_reference),
                    ]
                )

    sentence_reference_df = pd.DataFrame(
        sentence_reference_pairs, columns=["sentence", "reference"]
    )
    return sentence_reference_df


def preprocess_fast(df, nlp=None, label=None):
    sentences, references = [], []

    def matcher_df(text):
        for match in re.finditer(reference_pattern, text):
            start, end = match.span()
            sentence = text[: start - 1]
            reference = text[start:end]
            sentences.append(sentence)
            references.append(reference)

    df["text"] = df["text"].apply(lambda x: re.sub(r"[ ]+", " ", x.replace("\n", " ")))
    df["text"] = df["text"].apply(matcher_df)
    pairs = pd.DataFrame({"sentence": sentences, "reference": references})
    return pairs
