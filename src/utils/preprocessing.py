import os
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from spacy.language import Language
from tqdm import tqdm

from src.utils.common import load_spacy_model, read_csv
from src.utils.matcher import match_reference
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
    Language.component("reference_matcher", func=match_reference)

    nlp.add_pipe("sentencizer")
    nlp.add_pipe("reference_matcher", before="tagger")
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
        nlp = build_pipeline(
            disable=[
                "tagger",
                "parser",
                "ner",
                "tok2vec",
                "morphologizer",
                "attribute_ruler",
                "lemmatizer",
            ]
        )

    # Replacing newlines and multiple spaces
    df["text"] = df["text"].apply(lambda x: re.sub(r"[ ]+", " ", x.replace("\n", " ")))

    sentence_reference_pairs = []
    # def process_item(text):
    for item in tqdm(df["text"], desc=label):
        doc = nlp(item)

        for sentence in doc.sents:
            references = [
                ent for ent in sentence.ents if ent.label_ == "SECTION_REFERENCE"
            ]

            # Select sentences with only one reference
            # if len(references) == 1:
            for ref in references:
                sentence_reference_pairs.append(
                    [
                        sentence,
                        str(ref),
                    ]
                )
    # df["text"] = df["text"].apply(process_item)

    sentence_reference_df = pd.DataFrame(
        sentence_reference_pairs, columns=["sentence", "reference"]
    )
    return sentence_reference_df


def preprocess_fast(df, nlp=None, label=None):
    sentences, references, document_vectors = [], [], []

    def matcher_df(row):

        for match in re.finditer(reference_pattern, row["text"]):
            start, end = match.span()

            sentence = row["text"][max(0, start - 100) : start - 1]
            reference = row["text"][start:end]
            sentences.append(nlp(sentence))
            references.append(reference)
            docvect = {
                "verfahrensart": row["Verfahrensart"],
                "entscheidungsart": row["Entscheidungsart"],
            }
            document_vectors.append(docvect)

    df["text"] = df["text"].apply(lambda x: re.sub(r"[ ]+", " ", x.replace("\n", " ")))
    df = df.apply(matcher_df, axis=1)
    pairs = pd.DataFrame(
        {
            "sentence": sentences,
            "reference": references,
            "document_vector": document_vectors,
        }
    )

    return pairs


def load_vocab(data, classes_min_count=1):
    # a hacky way to get a onehot encoding for words from the target domain
    references = data["reference"].to_list()
    references = ["<start> " + ref + " <end>" for ref in references]
    references = [reference.split(" ") for reference in references]
    target_vocab = Word2Vec(
        sentences=references, vector_size=1, window=1, min_count=1
    ).wv

    sentences = data["sentence"].to_list()
    sentences = [("<start> " + sentence.text).split(" ") for sentence in sentences]
    input_vocab = Word2Vec(sentences=sentences, vector_size=1, window=1, min_count=1).wv

    # add empty token to output classes
    reference_data = data["reference"].to_list() + [
        "<empty>" for _ in range(classes_min_count)
    ]
    ref_classes = Word2Vec(
        sentences=[reference_data],
        vector_size=1,
        window=1,
        min_count=classes_min_count,
    ).wv
    ref_classes

    return input_vocab, target_vocab, ref_classes


def chunker(seq, size):
    """ returns batches of size size from sequence of objects"""
    len_cropped = len(seq) // size * size
    return (seq[pos : pos + size] for pos in range(0, len_cropped, size))


verfahrensarten = [
    "A",
    "AV",
    "B",
    "BN",
    "C",
    "CN",
    "D",
    "D-PKH",
    "DB",
    "DW",
    "F",
    "KSt",
    "P",
    "PB",
    "PKH",
    "VR",
    "WA",
    "WB",
    "WD",
    "WDB",
    "WDS-AV",
    "WDS-KSt",
    "WDS-PKH",
    "WDS-VR",
    "WDW",
    "WNB",
    "WRB",
]


def docvecs_from_chunk(chunk):
    verfahrensart_ids = []
    entscheidungsart_ids = []
    for j in range(len(chunk)):
        sample = chunk.iloc[j]
        verfahrensart_id = verfahrensarten.index(
            sample["document_vector"]["verfahrensart"]
        )
        entscheidungsart_id = (
            sample["document_vector"]["entscheidungsart"] == "B"
        )  # set to 1 if B or to 0 if U
        verfahrensart_ids.append(verfahrensart_id)
        entscheidungsart_ids.append(entscheidungsart_id)
    return np.array(verfahrensart_ids), np.array(entscheidungsart_ids)


def cnn_generator(
    data,
    target_vocab,
    target_vocab_size,
    input_vocab,
    input_vocab_size,
    sentence_len,
    batch_size,
    use_document_context,
):
    start_token = word2idx(input_vocab, "<start>", input_vocab_size)

    # loop through dataset endless times
    while True:
        # shuffle dataset
        data = data.sample(frac=1).reset_index(drop=True)

        # loop through dataset in batches
        for chunk in chunker(data, batch_size):
            ref_vectors = []
            sent_vectors = []

            # for all samples in batch
            for j in range(len(chunk)):

                sample = chunk.iloc[j]
                # process input
                sent_vectors.append(
                    process_input_sample(
                        input_vocab,
                        input_vocab_size,
                        sample["sentence"].text,
                        sentence_len,
                        start_token,
                    )
                )
                # process output
                ref_vectors.append(
                    word2onehot(target_vocab, sample["reference"], target_vocab_size)
                )

            ref_vectors = np.array(ref_vectors)
            sent_vectors = np.array(sent_vectors)
            verfahrensart_ids, entscheidungsart_ids = docvecs_from_chunk(chunk)
            if use_document_context:
                yield (
                    sent_vectors,
                    verfahrensart_ids,
                    entscheidungsart_ids,
                ), ref_vectors
            else:
                yield sent_vectors, ref_vectors


def seq2seq_generator(
    data,
    target_vocab,
    target_vocab_size,
    input_vocab,
    input_vocab_size,
    sentence_len,
    decoder_sentence_len,
    batch_size,
):
    end_token = word2onehot(target_vocab, "<end>", target_vocab_size)
    start_token = word2onehot(input_vocab, "<start>", input_vocab_size)
    # start_target_token = word2onehot(target_vocab, "<start>", target_vocab_size)
    start_target_token = "<start>"
    for sample in data.iloc:

        sent_vectors = process_input_sample(
            input_vocab,
            input_vocab_size,
            sample["sentence"].text,
            sentence_len,
            start_token,
        )

        ref_vectors = sample["reference"].split(" ")
        # crop to fixed number of input words
        ref_vectors = ref_vectors[
            : decoder_sentence_len - 1
        ]  # -1 because of start token

        # padding
        padding_target = decoder_sentence_len - 1 - len(ref_vectors)
        ref_vectors = (
            [start_target_token]
            + ref_vectors
            + ["<end>" for i in range(padding_target)]
        )

        # convert words to integers
        ref_vectors_ids = [
            word2idx(target_vocab, word, target_vocab_size) for word in ref_vectors
        ]
        # convert words to onehot encoded rows
        ref_vectors_onehots = [
            word2onehot(target_vocab, word, target_vocab_size) for word in ref_vectors
        ]

        ref_vectors_shifted = np.array(ref_vectors_onehots[1:] + [end_token])
        ref_vectors_ids = np.array(ref_vectors_ids)
        ref_vectors_shifted = np.array(ref_vectors_shifted)

        ref_vectors_ids = np.reshape(
            ref_vectors_ids, (batch_size, decoder_sentence_len)
        )
        ref_vectors_shifted = np.reshape(
            ref_vectors_shifted, (batch_size, decoder_sentence_len, target_vocab_size)
        )
        sent_vectors = np.reshape(sent_vectors, (batch_size, sentence_len))

        x = {"encoder_input": sent_vectors, "decoder_input": ref_vectors_ids}
        print("sent_vector shape:", sent_vectors.shape)
        print("ref_vector_ids:", ref_vectors_ids.shape)
        print("ref_vector_shifted:", ref_vectors_shifted.shape)
        yield x, ref_vectors_shifted


def process_input_sample(input_vocab, input_vocab_size, text, sentence_len, pad_token):
    """ performs padding, crops to fixed number of words and returns np.array"""
    sentence = text.split(" ")
    sent_vectors = [word2idx(input_vocab, word, input_vocab_size) for word in sentence]

    sent_vectors = sent_vectors[:sentence_len]
    # preppend with padding token if sentence is to short
    input_pad_length = sentence_len - len(sent_vectors)
    sent_vectors = [pad_token for i in range(input_pad_length)] + sent_vectors
    sent_vectors = np.array(sent_vectors)
    sent_vectors = np.reshape(sent_vectors, (sentence_len))
    return sent_vectors


def word2idx(vocab, word, tmp=None):
    if word not in vocab.index_to_key:
        return vocab.key_to_index["<empty>"]
    else:
        return vocab.key_to_index[word]


def idx2word(vocab, idx):
    try:
        return vocab.index_to_key[idx]
    except TypeError:
        print("key: ", idx)


def word2onehot(vocab, word, vocab_size):
    vect = np.zeros(vocab_size, dtype=np.uint8)
    vect[word2idx(vocab, word)] = 1
    return vect
