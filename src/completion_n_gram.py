"""
n-gram completion of references

This module implements the n-gram based reference completion.
"""

import re
from collections import Counter

from rich import box, print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.completion import Completion
from src.utils.common import timer
from src.utils.preprocessing import build_pipeline, preprocess
from src.utils.regular_expressions import word_pattern
from src.utils.stopwords import stopwords


def clean(sentence):
    tokens_to_remove = {"PUNCT", "NUM", "SYM"}
    return list(filter(lambda tok: tok.pos_ not in tokens_to_remove, sentence))


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


class NGramCompletion(Completion):
    """
    Class for n-gram completion of references
    """

    nlp = None
    stopwords = None
    sentence_reference_train = None
    sentence_reference_test = None
    sentence_reference_dev = None
    bigram_counts = None
    wordlist = None

    def __init__(self):
        Completion.__init__(self)
        self.nlp = build_pipeline(disable=["tagger", "parser", "ner"])
        self.stopwords = stopwords(self.nlp)

    @timer
    def feed_data(self, filename: str = "", key: str = ""):
        Completion.feed_data(self, filename=filename, key=key)

        print("\nFinding section references ...")

        self.sentence_reference_train = preprocess(
            self.data_train, nlp=self.nlp, label="training set"
        )
        self.sentence_reference_test = preprocess(
            self.data_test, nlp=self.nlp, label="test set"
        )
        self.sentence_reference_dev = preprocess(
            self.data_dev, nlp=self.nlp, label="dev set"
        )

        self.find_ngrams(self.sentence_reference_train)
        self.find_ngrams(self.sentence_reference_dev)
        self.find_ngrams(self.sentence_reference_test)

        self.find_bigrams(self.sentence_reference_train)
        self.find_bigrams(self.sentence_reference_dev)
        self.find_bigrams(self.sentence_reference_test, test=True)

    @timer
    def find_ngrams(self, df):
        df["cleaned"] = df["sentence"].apply(clean)
        df["ngram"] = df.apply(
            lambda row: find_ngram(
                sentence=row["cleaned"],
                reference=row["reference"],
                n=4,
                stopwords=self.stopwords,
            ),
            axis=1,
        )

    @timer
    def find_bigrams(self, df, test=False):
        bigrams = []

        if not self.wordlist:
            self.wordlist = []

        for ngram in tqdm(df["ngram"], desc="Building bigrams ..."):
            self.wordlist += ngram
            for i in range(len(ngram) - 1):
                bigrams.append((ngram[i], ngram[i + 1]))

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

        if not test:
            for bigram in tqdm(bigrams, desc="Counting bigrams ..."):
                secondaries = [b[1] for b in bigrams if b[0] == bigram[0]]
                self.bigram_counts[bigram[0]] = Counter(secondaries)

    def get_bigram_prob(self, bigram):
        try:
            counter = self.bigram_counts[bigram[0]]

            # Laplace smoothing (+1)
            prob = (counter[bigram[1]] + 1) / (sum(counter.values()) + 1)
            return prob

        # TODO: n-gram unknowns words
        except KeyError:
            return 1

    def get_ngram_prob(self, ngram):
        prob = 1
        for i in range(len(ngram) - 1):
            prob *= self.get_bigram_prob((ngram[i], ngram[i + 1]))
        return prob

    def get_suggestions(self, ngram):
        ngram = [x.lower() for x in ngram if x not in self.stopwords]
        probs = []
        for word in [word for word in self.wordlist if "§" in word]:
            prob = self.get_ngram_prob(ngram + [word])
            if prob != 0 and "§" in word:
                probs.append((prob, word))
        return sorted(probs)[::-1][:100]

    @timer
    def evaluate(self):
        first = 0
        three = 0
        incorrect = 0
        failed = 0

        print("\nEvaluating ...")
        for test_ngram in tqdm(
            self.sentence_reference_test["ngram"], desc="Evaluation"
        ):
            x = test_ngram[:-1]  # Input (n-1)-gram
            y = test_ngram[-1:][0]  # Output 1-gram

            # Top 3 suggestions
            suggestions = [suggestion[1] for suggestion in self.get_suggestions(x)[:3]]
            if len(suggestions) > 0:
                if y == suggestions[0]:
                    first += 1
                if y in suggestions:
                    three += 1
                else:
                    incorrect += 1
            else:
                failed += 1

        overall_count = three + incorrect + failed

        print()
        table = Table(
            title="Evaluation results:",
            title_justify="left",
            show_header=False,
            show_lines=False,
            box=box.ASCII_DOUBLE_HEAD,
        )
        table.add_row("overall test samples", f"{overall_count}")
        table.add_row(
            "[green]correct (first)", f"{first} ({first / overall_count:2.5f})"
        )
        table.add_row(
            "[yellow]correct (top 3)", f"{three} ({three / overall_count:2.5f})"
        )
        table.add_row(
            "[red]incorrect", f"{incorrect} ({incorrect / overall_count:2.5f})"
        )
        table.add_row("failed", f"{failed} ({failed / overall_count:2.5f})")
        print(table)
