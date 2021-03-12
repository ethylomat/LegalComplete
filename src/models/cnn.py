import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

import wandb
from src.utils.preprocessing import (
    cnn_generator,
    idx2word,
    preprocess_fast,
    process_input_sample,
    word2idx,
    word2onehot,
)

""" contains model with word2vec word embeddings as input and
Encoder-Decoder LSTM """


class CnnModel:
    def __init__(
        self, args, data_train, data_test, input_vocab, target_vocab, ref_classes
    ):
        self.target_vocab = target_vocab
        self.input_vocab = input_vocab
        self.ref_classes = ref_classes
        self.batch_size = 16
        self.epochs = 10
        self.max_sentence_len = 20
        self.max_decoder_seq_length = 9
        self.steps_per_epoch = len(data_train) // self.batch_size
        self.validation_steps = len(data_test) // self.batch_size
        self.window_size = 5
        self.latent_dim = 10

        """sentences = data_train["sentence"].to_list()
        sentences = [sentence.text.split(" ") for sentence in sentences]
        self.word_model = Word2Vec(
            sentences=sentences,
            vector_size=10,
            window=self.window_size,
            min_count=1,
            workers=4,
        )"""

        self.num_encoder_tokens = self.input_vocab.vectors.shape[0]
        self.num_decoder_tokens = self.ref_classes.vectors.shape[0]
        print("finished init setup")
        self.model = self.build_model()
        print("finished building model")

    def build_model(self):
        """ builds and returns encoder decoder tensorflow model"""

        inputs = tf.keras.Input(shape=(None,))
        x = layers.Embedding(self.num_encoder_tokens, self.latent_dim)(inputs)
        x = layers.Dropout(0.4)(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(16, return_sequences=False, return_state=False)
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        predictions = tf.keras.layers.Dense(
            self.num_decoder_tokens, activation="softmax"
        )(x)

        model = Model(inputs, predictions)
        model.summary()
        return model

    def preprocess(self, data):
        return cnn_generator(
            data,
            self.ref_classes,
            self.num_decoder_tokens,
            self.input_vocab,
            self.num_encoder_tokens,
            self.max_sentence_len,
            self.batch_size,
        )

    def train(self, train_df, eval_df):
        train_data = self.preprocess(train_df)
        eval_data = self.preprocess(eval_df)

        self.model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"]
        )
        wandb_callback = wandb.keras.WandbCallback(
            verbose=0,
            mode="auto",
            save_weights_only=False,
            log_weights=False,
            log_gradients=False,
            save_model=True,
            training_data=None,
            validation_data=None,
            labels=[],
            data_type=None,
            predictions=36,
            generator=None,
            input_type=None,
            output_type=None,
            log_evaluation=True,
            log_batch_frequency=None,
            log_best_prefix="best_",
        )

        self.model.fit(
            train_data,
            callbacks=[wandb_callback],
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=eval_data,
            epochs=self.epochs,
        )

    def predict(self, sent, beam_search_width):
        # TODO: implement multiple suggestions with beam_search_width param
        # for now only the first suggestion is made, others are set to None
        pad_token = word2idx(self.input_vocab, "<start>", self.num_encoder_tokens)
        sent = process_input_sample(
            self.input_vocab,
            self.num_encoder_tokens,
            sent,
            self.max_sentence_len,
            pad_token,
        )
        sent = np.reshape(
            sent,
            (
                1,
                -1,
            ),
        )  # reshape to batch_size 1
        prediction = self.model.predict(sent, batch_size=1)[0]

        # prediction_maxed = tf.argmax(prediction, axis=-1)[0]
        max_n = list(
            np.argpartition(prediction, -beam_search_width)[-beam_search_width:]
        )
        max_n.reverse()

        final_predicions = [idx2word(self.ref_classes, pred) for pred in max_n]
        return final_predicions

    def batch_predict(self, data, beam_search_width):
        results = []
        for el in tqdm(data.iloc):
            results.append(self.predict(el["sentence"].text, beam_search_width))
        return results

    def eval(self, eval_df):
        return self.model.eval(eval_df)
