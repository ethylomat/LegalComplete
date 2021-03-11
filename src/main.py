"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""
import argparse

from src.completion import Completion, print_metrics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dont_train", action="store_true", help="set to disable training"
    )
    parser.add_argument(
        "--dont_eval", action="store_true", help="set to disable evaluation"
    )
    parser.add_argument(
        "--evaluate_trigger",
        action="store_true",
        help="set to enable trigger evaluation after NGRAM training",
    )
    parser.add_argument(
        "--model_name", type=str, help="set model name eg. seq2seq or ngram"
    )
    parser.add_argument(
        "--dataset", type=str, default="test_ce_bverwg", help="dataset path or key"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    c = Completion(args)
    if not args.dont_train:
        c.train_data()
    if not args.dont_eval:
        metrics = c.evaluate_references(c.data_test)
        print_metrics(metrics)

    if args.evaluate_trigger:
        assert (
            args.model_name == "NGRAM"
        ), "trigger evaluation is only defined when using --model_name==NGRAM"
        c.evaluate_trigger(c.data_test)
