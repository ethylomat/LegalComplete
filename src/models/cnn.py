import time

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
    chunker,
    cnn_generator,
    docvecs_from_chunk,
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
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.max_sentence_len = args.input_sentence_len
        self.max_decoder_seq_length = 9
        self.steps_per_epoch = len(data_train) // self.batch_size
        self.validation_steps = len(data_test) // self.batch_size
        self.args = args

        self.latent_dim = args.embedding_dim
        self.num_encoder_tokens = self.input_vocab.vectors.shape[0]
        self.num_decoder_tokens = self.ref_classes.vectors.shape[0]

        self.pad_token = word2idx(self.input_vocab, "<start>", self.num_encoder_tokens)

        print("finished init setup")
        if not args.weights:
            self.model = self.build_model()

            print("finished building model")
        else:
            self.model = tf.keras.models.load_model(args.weights)
            print("finished loading model weights")

    def build_model(self):
        """ builds and returns encoder decoder tensorflow model"""

        inputs = tf.keras.Input(shape=(None,))
        x = layers.Embedding(self.num_encoder_tokens, self.latent_dim)(inputs)
        x = layers.Dropout(0.4)(x)
        if self.args.use_GRU:
            x = layers.GRU(64, return_sequences=False, return_state=False)(x)
        else:
            x = layers.LSTM(64, return_sequences=False, return_state=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        if self.args.use_document_context:
            verfahrensart_input = tf.keras.Input(shape=(1,))
            dokumentart_input = tf.keras.Input(shape=(1,))
            concat_list = [x, verfahrensart_input, dokumentart_input]
            x = tf.keras.layers.Concatenate()(concat_list)

        x = tf.keras.layers.Dense(255, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_decoder_tokens, activation="softmax")(x)
        inputs_list = [inputs]
        if self.args.use_document_context:
            inputs_list.append(verfahrensart_input)
            inputs_list.append(dokumentart_input)
        model = Model(inputs_list, outputs)
        tf.keras.utils.plot_model(
            model, "RNN_model.png", show_shapes=True, show_layer_names=False
        )
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
            self.args.use_document_context,
        )

    def train(self, train_df, eval_df):
        train_data = self.preprocess(train_df)
        eval_data = self.preprocess(eval_df)
        topn_acc = tf.keras.metrics.TopKCategoricalAccuracy(
            k=3, name="top_k_categorical_accuracy", dtype=None
        )

        self.model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["acc", topn_acc],
        )

        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            "weights/" + str(time.time()) + "model_weights.h5",
            save_weights_only=False,
            save_best_only=True,
            monitor="val_acc",
            mode="max",
        )
        callbacks = [mcp_save]
        if not self.args.nowandb:
            wandb_callback = wandb.keras.WandbCallback(
                verbose=0,
                mode="auto",
                save_weights_only=False,
                log_weights=False,
                log_gradients=False,
                save_model=True,
                log_evaluation=True,
                log_best_prefix="best_",
            )
            callbacks.append(wandb_callback)

        self.model.fit(
            train_data,
            callbacks=callbacks,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=eval_data,
            epochs=self.epochs,
        )

    def predict(
        self,
        batch,
        verfahrensart_ids,
        entscheidungsart_ids,
        beam_search_width,
        batch_size,
    ):
        # preprocess samples in batch
        batch = np.array(
            [
                process_input_sample(
                    self.input_vocab,
                    self.num_encoder_tokens,
                    sample.text,
                    self.max_sentence_len,
                    self.pad_token,
                )
                for sample in batch
            ]
        )

        batch = np.reshape(
            batch,
            (
                batch_size,
                -1,
            ),
        )  # reshape to batch_size
        if self.args.use_document_context:
            model_inputs = [batch, verfahrensart_ids, entscheidungsart_ids]
        else:
            model_inputs = batch
        batch_prediction = self.model.predict(model_inputs, batch_size=batch_size)

        batch_top_n = []
        batch_top_probabilities = []
        for prediction in batch_prediction:
            # argmax of top n arguments
            indicies = np.argpartition(prediction, -beam_search_width)[
                -beam_search_width:
            ]
            indicies_sorted = indicies[np.argsort(prediction[indicies])][::-1]
            batch_top_n.append(indicies_sorted)
            batch_top_probabilities.append(prediction[indicies_sorted])

        batch_top_n_words = []
        # convert back from ids to words
        for top_n in batch_top_n:
            batch_top_n_words.append(
                [idx2word(self.ref_classes, pred) for pred in top_n]
            )
        return batch_top_n_words, batch_top_probabilities

    def batch_predict(self, data, beam_search_width):

        batch_words = []
        batch_probabilities = []
        for chunk in chunker(data, self.batch_size):
            # loop through dataset in batches
            sample = chunk["sentence"].to_list()
            verfahrensart_ids, entscheidungsart_ids = docvecs_from_chunk(chunk)

            words, probabilities = self.predict(
                sample,
                verfahrensart_ids,
                entscheidungsart_ids,
                beam_search_width,
                self.batch_size,
            )
            batch_words += list(words)
            batch_probabilities += list(probabilities)
        return batch_words, batch_probabilities
