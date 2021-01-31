"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""
import argparse

from src.completion import (
    Completion,
    evaluate_references,
    evaluate_trigger,
    print_metrics,
)


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
        metrics = evaluate_references(c.data_test, c.refmodel, c.nlp)
        print_metrics(metrics)
        evaluate_trigger(c.data_test, c.refmodel, c.nlp)
