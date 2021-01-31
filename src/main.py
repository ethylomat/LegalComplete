import argparse

from src.completion import Completion, call_evaluate, print_metrics

"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dont_train", action="store_true", help="set to disable training"
    )
    parser.add_argument(
        "--dont_eval", action="store_true", help="set to disable evaluation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    c = Completion()
    c.feed_data(key="test_ce_bverwg")
    if not args.dont_train:
        c.train_data()
    if not args.dont_eval:
        metrics = call_evaluate(c.data_test, c.refmodel, c.nlp)
        print_metrics(metrics)
