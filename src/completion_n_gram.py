"""
n-gram completion of references

This module implements the n-gram based reference completion.
"""

import re
from collections import Counter
from typing import List

import pandas as pd
from tqdm import tqdm

from src.utils.common import timer
from src.utils.regular_expressions import word_pattern
from src.utils.stopwords import stopwords


def clean(sentence):
    """
    Function to clean punctuation, numbers and symbols from SpaCy sentences.
    Is applied to Pandas dataframe.

    Args:
        sentence: input sentence (SpaCy doc)
    Returns:
        cleaned: list of words (not SpaCy doc) not containing punctuation, numbers, symbols
    """
    tokens_to_remove = {"PUNCT", "NUM", "SYM"}
    return [word for word in sentence if word.pos_ not in tokens_to_remove]


def clean_ngram(ngram):
    lower = [tok.text.lower() for tok in ngram]
    filter_words = [word for word in lower if re.search(word_pattern, word)]
    return filter_words


def find_ngram(sentence, reference, n: int, stopwords):
    for i, word in enumerate(sentence):
        if word.ent_type_ == "SECTION_REFERENCE" and word.ent_iob_ == "B":
            try:
                sentence_start = sentence[:i]
                ngram = [
                    word
                    for word in sentence_start
                    if word.text.lower() not in stopwords
                ]
                return clean_ngram(ngram)[-n:] + [reference]

            # Reference in the beginning of the line
            except IndexError:
                return None


def find_norm_bigram(sentence, reference):
    for i, word in enumerate(sentence):
        if word.ent_type_ == "SECTION_REFERENCE" and word.ent_iob_ == "B":
            try:
                sentence_start = sentence[:i]
                ngram = [word for word in sentence_start]
                return clean_ngram(ngram)[-1:] + [reference]

            # Reference in the beginning of the line
            except IndexError:
                return None


class NGramCompletion:
    """
    Class for n-gram completion of references
    """

    data_train = None
    data_dev = None
    nlp = None
    stopwords = None
    sentence_reference_train = None
    data_test = None
    sentence_reference_dev = None
    bigram_counts = None
    bigram_reference_probs = None
    wordlist = None

    def __init__(self, nlp):
        self.nlp = nlp
        self.stopwords = stopwords(self.nlp)

    @timer
    def train(self, data_train, data_dev):
        self.sentence_reference_train = data_train
        self.sentence_reference_dev = data_dev
        print("\nFinding section references ...")

        self.find_ngrams(self.sentence_reference_train)
        self.find_ngrams(self.sentence_reference_dev)

        self.find_bigrams(self.sentence_reference_train)
        self.find_bigrams(self.sentence_reference_dev, test=True)

    @timer
    def batch_predict(
        self, data: pd.DataFrame, num_suggestions: int
    ) -> List[List[str]]:
        # prepare eval data
        self.find_ngrams(data)
        self.find_bigrams(data, test=True)
        batch_suggestions = []
        for test_sample in data.iloc:
            # trigger_prob = self.get_trigger_prob(test_sample["ngram no sw"][-2:-1])
            x = test_sample["ngram"][:-1]  # Input (n-1)-gram
            # y = test_sample["ngram"][-1:][0]  # Output 1-gram

            # Top 3 suggestions
            suggestions = [suggestion[1] for suggestion in self.get_suggestions(x, 3)]
            batch_suggestions.append(suggestions)
        return batch_suggestions

    @timer
    def find_ngrams(self, df):
        df["cleaned"] = df["sentence"].apply(clean)

        # n-grams to determine norm reference
        df["ngram"] = df.apply(
            lambda row: find_ngram(
                sentence=row["cleaned"],
                reference=row["reference"],
                n=4,
                stopwords=self.stopwords,
            ),
            axis=1,
        )

        # n-grams to determine the presence of norm reference
        df["ngram no sw"] = df.apply(
            lambda row: find_ngram(
                sentence=row["cleaned"],
                reference=row["reference"],
                n=5,
                stopwords=[],
            ),
            axis=1,
        )

    @timer
    def find_bigrams(self, df, test=False):
        # TODO: Refactoring (-> DRY, with/without sw)

        bigrams = []
        bigrams_no_sw = []

        if not self.wordlist:
            self.wordlist = []
        for ngram in tqdm(df["ngram"], desc="Building bigrams (with stopwords) ..."):
            self.wordlist += ngram
            for i in range(len(ngram) - 1):
                bigrams.append((ngram[i], ngram[i + 1]))

        for ngram in tqdm(
            df["ngram no sw"], desc="Building bigrams (without stopwords) ..."
        ):
            self.wordlist += ngram
            for i in range(len(ngram) - 1):
                bigrams_no_sw.append((ngram[i], ngram[i + 1]))

        self.wordlist = list(set(self.wordlist))

        """
        Example of bigram_counts dictionary:

        {
            'kostenentscheidung':
                Counter({
                    '§ 154 abs. 2 vwgo': 20,
                    '§ 155 abs. 1 satz 3 vwgo': 1,
                    'ergibt': 1
                })
        }
        """

        if not self.bigram_counts:
            self.bigram_counts = {}

        if not self.bigram_reference_probs:
            self.bigram_reference_probs = {}

        if not test:
            # Initializing dict of bigrams with
            # key: first item of bigrams
            # item: list of secondary items of bigrams
            bigram_dict = {bigram[0]: [] for bigram in bigrams}
            for bigram in tqdm(bigrams, desc="Counting bigrams (with stopwords) ..."):
                bigram_dict[bigram[0]].append(bigram[1])

            bigrams_no_sw_dict = {bigram[0]: [] for bigram in bigrams_no_sw}
            for bigram in tqdm(
                bigrams_no_sw, desc="Counting bigrams (without stopwords) ..."
            ):
                bigrams_no_sw_dict[bigram[0]].append(bigram[1])

            # Applying Count dictionary (collections.Count()) on secondary items list
            for first, secondaries in bigram_dict.items():
                self.bigram_counts[first] = Counter(secondaries)

            for first, secondaries in bigrams_no_sw_dict.items():
                reference_count = len(
                    [
                        secondary
                        for secondary in secondaries
                        if secondary.startswith("§")
                    ]
                )
                no_reference_count = len(
                    [
                        secondary
                        for secondary in secondaries
                        if not secondary.startswith("§")
                    ]
                )

                try:
                    prob = reference_count / no_reference_count
                except ZeroDivisionError:
                    prob = 0

                self.bigram_reference_probs[first] = {
                    "ref": reference_count,
                    "noref": no_reference_count,
                    "prob": prob,
                }

    def get_bigram_prob(self, bigram):
        try:
            counter = self.bigram_counts[bigram[0]]

            # Laplace smoothing (+1)
            prob = (counter[bigram[1]] + 1) / (sum(counter.values()) + 1)
            return prob

        # No effect for unknown words
        except KeyError:
            return 1

    def get_ngram_prob(self, ngram):
        prob = 1
        for i in range(len(ngram) - 1):
            prob *= self.get_bigram_prob((ngram[i], ngram[i + 1]))
        return prob

    def get_suggestions(self, ngram, num_suggestions: int):
        """
        Method to receive n-gram based norm reference suggestions.

        Args:
            ngram: n-1 gram input for suggestion
            num_suggestions: only the top n suggestions will be returned
        Returns:
            probs: sorted list of top (probability, word) pairs for suggestion.
            Probability is currently not normed.
        """
        ngram = [x.lower() for x in ngram if x not in self.stopwords]
        probs = []
        for word in [word for word in self.wordlist if "§" in word]:
            prob = self.get_ngram_prob(ngram + [word])
            if prob != 0 and "§" in word:
                probs.append((prob, word))
        probs = sorted(probs)[::-1][:num_suggestions]
        return probs

    def get_trigger_prob(self, ngram):
        """
        Method to receive probability to trigger suggestion event.

        Args:
            ngram: n-1 gram input for suggestion probability
        Returns:
            prob: probability of triggering suggestion
            (probability of section references for bigram)
        """
        ngram = [x.lower() for x in ngram]

        if len(ngram) > 0:
            last_word = ngram[-1]
        else:
            return 0

        if last_word.startswith("§"):
            return 1
        try:
            prob = self.bigram_reference_probs[last_word]["prob"]
        except KeyError:
            prob = 0
        return prob
