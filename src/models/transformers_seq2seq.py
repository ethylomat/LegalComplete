import logging

import pandas as pd
import torch
from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel

from src.utils.preprocessing import preprocess_fast


class TransSeqModel:
    def __init__(self, args):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        model_args = Seq2SeqArgs()
        model_args.num_train_epochs = 1
        model_args.no_save = True
        model_args.train_batch_size = 4
        model_args.evaluate_generated_text = True
        model_args.evaluate_during_training = False
        model_args.evaluate_generated_text = False
        model_args.evaluate_during_training_verbose = True
        model_args.use_multiprocessing = False
        model_args.max_seq_length = 5
        model_args.max_length = 6
        model_args.overwrite_output_dir = True
        self.model_args = model_args
        cuda_available = torch.cuda.is_available()

        # Initialize model
        self.model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name="facebook/bart-base",
            args=model_args,
            use_cuda=cuda_available,
        )

    def count_matches(labels, preds):
        return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])

    def preprocess(self, data):
        data = data.filter(["sentence", "reference"])
        data = data.rename(
            columns={"sentence": "input_text", "reference": "target_text"}
        )
        data = data.apply(str)
        return data

    def train(self, train_df, eval_df):
        train_df = self.preprocess(train_df)
        eval_df = self.preprocess(eval_df)
        self.model.train_model(train_df, eval_data=eval_df, matches=self.count_matches)

    def predict(self, sent, beam_search_width):
        # TODO: implement multiple suggestions with beam_search_width param
        # for now only the first suggestion is made, others are set to None
        return self.model.predict(sent)[:beam_search_width]

    def batch_predict(self, data, beam_search_width):
        results = []
        for el in data.iloc:
            print("sent is:", el["sentence"].text)
            print(type(el["sentence"].text), "was type")
            results.append(self.predict(el["sentence"].text, beam_search_width))
            print(el["sentence"], "->", results[-1])
        return results

    def eval(self, eval_df):
        return self.model.eval(eval_df)
