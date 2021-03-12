"""
Completion base class

The completion base class is used to build completion classes for
different completion methods (e.g. completion_n_gram.py). It stores
the datasets as dataframes and keeps training, development and test
subsets.
"""
from typing import Dict

import numpy as np
import pandas as pd
from rich import box, print
from rich.table import Table
from tqdm import tqdm

from src.completion_n_gram import NGramCompletion
from src.models.cnn import CnnModel
from src.models.lstm import LstmModel
from src.models.transformers_seq2seq import TransSeqModel
from src.utils.common import read_csv, split_dataframe
from src.utils.preprocessing import (
    build_pipeline,
    load_vocab,
    preprocess,
    preprocess_fast,
)
from src.utils.retrieve import download_dataset, get_dataset_info


class Completion:
    """
    All models can be trained and evaluated by
    creating an completion object with the respective model variant
    """

    def __init__(self, args):
        self.nlp = build_pipeline(
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
        self.feed_data(args, key=args.dataset)
        if args.model_name == "NGRAM":
            self.refmodel = NGramCompletion(self.nlp)
        elif args.model_name == "SEQ2SEQ":
            self.refmodel = TransSeqModel(args)
        elif args.model_name == "CNN":
            self.refmodel = CnnModel(
                args,
                self.data_train,
                self.data_test,
                self.input_vocab,
                self.target_vocab,
                self.ref_classes,
            )
        elif args.model_name == "LSTM":
            self.refmodel = LstmModel(
                args, self.data_train, self.input_vocab, self.target_vocab
            )
        else:
            raise ValueError("no model with this key available: ", args.model_name)

    def feed_data(self, args, filename: str = "", key: str = ""):
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

        full_df = read_csv(filename)
        if args.model_name == "NGRAM":
            self.preprocess = preprocess
        else:
            self.preprocess = preprocess_fast
        # full_df.drop(
        #     full_df.index[: len(full_df) // 2], 0, inplace=True
        # )  # reduce dataset for fast debugging TODO Remove

        full_df = self.preprocess(full_df, nlp=self.nlp, label="full df")
        self.input_vocab, self.target_vocab, self.ref_classes = load_vocab(full_df)

        # Splitting dataframe into different sets
        self.data_train, self.data_test, self.data_dev = split_dataframe(
            full_df, fracs=[0.80, 0.10, 0.10]
        )

    def train_data(self):
        """self.data_train = self.preprocess(
            self.data_train, nlp=self.nlp, label="training set"
        )
        self.data_dev = self.preprocess(self.data_dev, nlp=self.nlp, label="dev set")"""
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

    def print_outputs(self, data_test: pd.DataFrame, count: int):
        """ method for visualization of model output"""
        batch_suggestions, probabilities = self.refmodel.batch_predict(data_test, 3)
        i = 0
        for (suggestions, sample_probabilites, test_sample) in zip(
            batch_suggestions, probabilities, data_test.iloc
        ):
            y = test_sample["reference"]
            x = test_sample["sentence"].text
            print("input: ", x)
            print("ground truth: ", y)
            for sugg, prob in zip(suggestions, sample_probabilites):
                print("suggestion: ", prob, ": ", sugg)
            print()
            if i > count:
                break
            i += 1

    def evaluate_ROC(self, data_test: pd.DataFrame):
        import matplotlib
        from sklearn.metrics import auc, roc_curve

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        words, probabilities = self.refmodel.batch_predict(data_test, 1)
        # is list of lists with only one element
        probabilities = [p[0] for p in probabilities]
        words = [w[0] for w in words]
        y_true = np.array(data_test["reference"])

        # binary array whether prediction was true or false
        x_test = np.array([prediction == gt for prediction, gt in zip(words, y_true)])

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(x_test, probabilities)
        auc = auc(fpr_keras, tpr_keras)
        plt.plot(thresholds_keras, label="thresholds")
        plt.show()
        plt.plot(fpr_keras, tpr_keras, label="ROC")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve. AUC: " + str(auc))
        plt.show()

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

        print("\nEvaluating ...")
        batch_suggestions, _ = self.refmodel.batch_predict(data_test, 3)

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
            "first": first / overall_count,
            "three": three / overall_count,
            "incorrect": incorrect / overall_count,
            "failed": failed / overall_count,
        }
        return metrics

    def evaluate_trigger(self, data_test):

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
    table.add_row(
        "[green]correct (first)", f"{int(first * overall_count)} ({first:2.5f})"
    )
    table.add_row(
        "[yellow]correct (top 3)", f"{int(three * overall_count)} ({three:2.5f})"
    )
    table.add_row(
        "[red]incorrect", f"{int(incorrect * overall_count)} ({incorrect:2.5f})"
    )
    table.add_row("failed", f"{int(failed * overall_count)} ({failed:2.5f})")

    table.add_row("", "")

    print(table)
