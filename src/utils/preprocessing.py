import os
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
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

            sentence = text[max(0, start - 100) : start - 1]
            reference = text[start:end]
            sentences.append(nlp(sentence))
            references.append(reference)

    df["text"] = df["text"].apply(lambda x: re.sub(r"[ ]+", " ", x.replace("\n", " ")))
    df["text"] = df["text"].apply(matcher_df)
    pairs = pd.DataFrame({"sentence": sentences, "reference": references})
    return pairs


def load_vocab(data):
    # a hacky way to get a onehot encoding for words from the target domain
    references = data["reference"].to_list()
    references = [ref + " <end>" for ref in references]
    references = [reference.split(" ") for reference in references]
    target_vocab = Word2Vec(
        sentences=references, vector_size=1, window=1, min_count=1
    ).wv

    sentences = data["sentence"].to_list()
    sentences = [sentence.text.split(" ") for sentence in sentences]
    input_vocab = Word2Vec(sentences=sentences, vector_size=1, window=1, min_count=1).wv
    return input_vocab, target_vocab


def seq2seq_generator(
    data, target_vocab, target_vocab_size, input_vocab, input_vocab_size
):
    sentence_len = 7
    decoder_sentence_len = 7
    target_tensor_shape = (decoder_sentence_len, target_vocab_size)
    end_token = word2onehot(target_vocab, "<end>", target_vocab_size)
    for sample in data.iloc:

        sentence = sample["sentence"].text.split(" ")
        sent_vectors = [
            word2onehot(input_vocab, word, input_vocab_size) for word in sentence
        ]
        # crop to fixed number of input words
        sent_vectors = sent_vectors[: min(len(sent_vectors), sentence_len)]
        # preppend with padding token if sentence is to short
        sent_vectors = np.array(sent_vectors)

        reference = sample["reference"].split(" ")
        ref_vectors = [
            word2onehot(target_vocab, word, target_vocab_size) for word in reference
        ]

        padding_target = decoder_sentence_len - min(
            len(ref_vectors), decoder_sentence_len
        )

        ref_vectors_shifted = ref_vectors[1:] + [
            end_token for i in range(padding_target + 1)
        ]
        ref_vectors = ref_vectors + [end_token for i in range(padding_target)]

        ref_vectors = np.array(ref_vectors)
        ref_vectors_shifted = np.array(ref_vectors_shifted)

        ref_vectors_shifted = np.reshape(ref_vectors_shifted, target_tensor_shape)
        sent_vectors = np.reshape(sent_vectors, (sentence_len, input_vocab_size))
        print("shape is ", ref_vectors.shape, target_vocab_size)
        print("shape is ", ref_vectors_shifted.shape, target_vocab_size)
        x = {"encoder_input": sent_vectors, "decoder_input": ref_vectors}
        yield (x, ref_vectors_shifted)


def word2idx(vocab, word, tmp=None):
    try:
        return vocab.key_to_index[word]
    except KeyError:
        return vocab.key_to_index["."]


def idx2word(vocab, idx):
    return vocab.index_to_key[idx]


def word2onehot(vocab, word, vocab_size):
    vect = np.zeros(vocab_size, dtype=np.uint8)
    vect[word2idx(vocab, word)] = 1
    return vect
