"""
Completion base class

The completion base class is used to build completion classes for
different completion methods (e.g. completion_n_gram.py). It stores
the datasets as dataframes and keeps training, development and test
subsets.
"""
from typing import Dict

import pandas as pd
from rich import box, print
from rich.table import Table
from tqdm import tqdm

from src.completion_n_gram import NGramCompletion
from src.models.transformers_seq2seq import TransSeqModel
from src.utils.common import read_csv, split_dataframe
from src.utils.preprocessing import build_pipeline, preprocess, preprocess_fast
from src.utils.retrieve import download_dataset, get_dataset_info


class Completion:
    """
    Completion base class
    """

    def __init__(self, args):

        self.nlp = build_pipeline(disable=["tagger", "parser", "ner"])
        if args.model_name == "NGRAM":
            self.refmodel = NGramCompletion(self.nlp)
            self.preprocess = preprocess_fast
        elif args.model_name == "SEQ2SEQ":
            self.refmodel = TransSeqModel(args)
            self.preprocess = preprocess_fast
        else:
            raise ValueError("no model with this key available: ", args.model_name)

    def feed_data(self, filename: str = "", key: str = ""):
        """
        Method for reading raw files into datasets

        Args:
            filename: relative filename in data directory
            key: alternative - give dataset key to receive automatically
        """

        # If key is provided instead of filename (-> filename is overwritten)
        if key:
            dataset_info = get_dataset_info(key)
            download_dataset(dataset_info)
            filename = dataset_info["extracted"]

        # TODO: Catch FileNotFoundError
        full_df = read_csv(filename)

        # Splitting dataframe into different sets
        self.data_train, self.data_dev, self.data_test = split_dataframe(
            full_df, fracs=[0.80, 0.10, 0.10]
        )

    def train_data(self):
        self.data_train = self.preprocess(
            self.data_train, nlp=self.nlp, label="training set"
        )
        self.data_dev = self.preprocess(self.data_dev, nlp=self.nlp, label="dev set")
        self.refmodel.train(self.data_train, self.data_dev)

    def __str__(self):
        """
        String representation of completion objects.
        """
        class_name = str(self.__module__).split(".")[-1]
        return "<%s, Dataset: TRAIN %d | TEST %d | DEV %d>" % (
            class_name,
            self.data_train.shape[0],
            self.data_test.shape[0],
            self.data_dev.shape[0],
        )

    def evaluate_references(self, data_test: pd.DataFrame) -> Dict:
        """
        Method for evaluation of suggestions.
        Prints the amounts of correct suggestions based on the test set.
        Considering the suggestion being suggested at the top 3 suggestions.
        Args:
          data_test: data for evaluation
        Returns:
          metrics: dict containing first, three, incorrect, overall_count and more
        """

        first = 0
        three = 0
        incorrect = 0
        failed = 0

        data_test = self.preprocess(data_test, nlp=self.nlp, label="eval data")

        print("\nEvaluating ...")
        batch_suggestions = self.refmodel.batch_predict(data_test, 3)

        # compute metrics
        for (suggestions, test_sample) in zip(batch_suggestions, data_test.iloc):
            y = test_sample["reference"]
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
        metrics = {
            "overall_count": overall_count,
            "first": first,
            "three": three,
            "incorrect": incorrect,
            "failed": failed,
        }
        return metrics

    def evaluate_trigger(self, data_test):

        data_test = self.preprocess(data_test, nlp=self.nlp, label="test set")
        self.refmodel.find_ngrams(data_test)
        self.refmodel.find_bigrams(data_test, test=True)
        correct_trigger = 0
        false_trigger = 0
        TRIGGER_THRESHOLD = 0.9
        for test_sample in data_test.iloc:
            sample = test_sample["ngram no sw"]
            bigrams = zip(sample[:-1], sample[1:])
            for bigram in bigrams:
                x = bigram[:1]
                y = bigram[-1:]

                trigger_prob = self.refmodel.get_trigger_prob(x)
                if trigger_prob >= TRIGGER_THRESHOLD:
                    if y[0].startswith("§"):
                        correct_trigger += 1
                    else:
                        false_trigger += 1
        overall_trigger = correct_trigger + false_trigger
        table = Table(
            title="Evaluation results:",
            title_justify="left",
            show_header=False,
            show_lines=False,
            box=box.ASCII_DOUBLE_HEAD,
        )
        if overall_trigger == 0:
            raise Exception("trigger never triggered")
        else:
            table.add_row(
                "correct triggered",
                f"{correct_trigger} ({correct_trigger / overall_trigger:2.5f})",
            )
            table.add_row(
                "false triggered",
                f"{false_trigger} ({false_trigger / overall_trigger:2.5f})",
            )
            table.add_row(
                "overall triggered",
                f"{overall_trigger} ({overall_trigger / overall_trigger:2.5f})",
            )
            print(table)


def print_metrics(metrics):
    overall_count = metrics["overall_count"]
    first = metrics["first"]
    three = metrics["three"]
    incorrect = metrics["incorrect"]
    failed = metrics["failed"]

    # Printing the results
    print()
    table = Table(
        title="Evaluation results:",
        title_justify="left",
        show_header=False,
        show_lines=False,
        box=box.ASCII_DOUBLE_HEAD,
    )
    table.add_row("overall test samples", f"{overall_count}")
    table.add_row("[green]correct (first)", f"{first} ({first / overall_count:2.5f})")
    table.add_row("[yellow]correct (top 3)", f"{three} ({three / overall_count:2.5f})")
    table.add_row("[red]incorrect", f"{incorrect} ({incorrect / overall_count:2.5f})")
    table.add_row("failed", f"{failed} ({failed / overall_count:2.5f})")

    table.add_row("", "")

    print(table)
