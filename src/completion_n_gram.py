import re

from src.completion import Completion
from src.utils.preprocessing import build_pipeline, preprocess
from src.utils.regular_expressions import word_pattern
from src.utils.stopwords import stopwords

"""
Class for n-gram completion of references
"""


def clean(sentence):
    tokens_to_remove = {"PUNCT", "NUM", "SYM"}
    return list(filter(lambda tok: tok.pos_ not in tokens_to_remove, sentence))


def clean_ngram(ngram):
    lower = [tok.text.lower() for tok in ngram]
    filter_words = [word for word in lower if re.search(word_pattern, word)]
    return filter_words


def find_ngram(sentence, n: int, stopwords):
    for i, word in enumerate(sentence):
        if word.ent_type_ == "SECTION_REFERENCE" and word.ent_iob_ == "B":
            try:
                sentence_start = sentence[:i]
                sentence_start_removed_sw = [
                    word
                    for word in sentence_start
                    if word.text.lower() not in stopwords
                ]
                ngram = sentence_start_removed_sw
                return clean_ngram(ngram)[-n:]

            # Reference in the beginning of the line
            except IndexError:
                return None


class NGramCompletion(Completion):
    nlp = None
    stopwords = None
    sentence_reference_train = None
    sentence_reference_test = None
    sentence_reference_dev = None

    def __init__(self):
        Completion.__init__(self)
        self.nlp = build_pipeline()
        self.stopwords = stopwords(self.nlp)

    def feed_data(self, filename: str = "", key: str = ""):
        Completion.feed_data(self, filename=filename, key=key)
        # self.sentence_reference_train = preprocess(
        #    self.data_train,
        #    nlp=self.nlp,
        #    label="training set"
        # )
        self.sentence_reference_test = preprocess(
            self.data_test, nlp=self.nlp, label="test set"
        )
        self.sentence_reference_dev = preprocess(
            self.data_dev, nlp=self.nlp, label="dev set"
        )
        # self.find_ngrams(self.sentence_reference_train)
        self.find_ngrams(self.sentence_reference_test)
        self.find_ngrams(self.sentence_reference_dev)

    def find_ngrams(self, df):
        df["cleaned"] = df["sentence"].apply(clean)
        df["ngram"] = df["cleaned"].apply(find_ngram, n=4, stopwords=self.stopwords)
