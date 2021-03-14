"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""
import argparse

import wandb
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
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="number of training and eval batch size",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=32, help="vector length of word embeddings"
    )
    parser.add_argument(
        "--classes_min_count",
        type=int,
        default=1,
        help="when using RNNCLASS model mininum "
        "count for reference to be part of target classes",
    )
    parser.add_argument(
        "--input_sentence_len",
        type=int,
        default=30,
        help="number of words to input to model",
    )
    parser.add_argument(
        "--drop_data_rate",
        type=float,
        default=0.0,
        help="dataset samples will be dropped randomly according at this rate before preprocessing",
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="itialize model from this path"
    )
    parser.add_argument(
        "--nowandb", action="store_true", help="set to disable wandb connection"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    c = Completion(args)
    wandb_is_initialized = False

    if not args.dont_train:
        if not args.nowandb:
            wandb.init(project="legalcomplete", config=args)
            wandb_is_initialized = True
        c.train_data()
    if not args.dont_eval:
        metrics = c.evaluate_references(c.data_test)
        c.print_outputs(c.data_test, 5)
        if wandb_is_initialized:
            wandb.log({"metrics": metrics})
        print_metrics(metrics)
        c.evaluate_ROC(c.data_test)

    if args.evaluate_trigger:
        assert (
            args.model_name == "NGRAM"
        ), "trigger evaluation is only defined when using --model_name==NGRAM"
        c.evaluate_trigger(c.data_test)
