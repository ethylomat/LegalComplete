
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils.preprocessing import (
    idx2word,
    preprocess_fast,
    process_input_sample,
    cnn_generator,
    cnn_preprocessing,
    word2idx,
    word2onehot,
)

""" contains model with word2vec word embeddings as input and
Encoder-Decoder LSTM """


class CnnModel:
    def __init__(self, args, data_train, input_vocab, target_vocab, ref_classes):
        self.target_vocab = target_vocab
        self.input_vocab = input_vocab
        self.ref_classes = ref_classes
        self.batch_size = 1
        self.epochs = 2
        self.max_sentence_len = 12
        self.max_decoder_seq_length = 9

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
        x = layers.Embedding(self.num_encoder_tokens,
                self.latent_dim)(inputs)

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False, 
            return_state=False))(x)
        x = layers.SpatialDropout1D(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        predictions = tf.keras.layers.Dense(self.num_decoder_tokens, activation="softmax")(x)

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
        )

    def train(self, train_df, eval_df):
        train_data = self.preprocess(train_df)
        eval_data = self.preprocess(eval_df)

        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        self.model.fit(
            train_data,
            steps_per_epoch=5,
            validation_data=eval_data,
            epochs=self.epochs,
        )

    def predict(self, sent, beam_search_width):
        # TODO: implement multiple suggestions with beam_search_width param
        # for now only the first suggestion is made, others are set to None
        pad_token = word2idx(self.input_vocab, "<start>", self.num_encoder_tokens)
        sent = process_input_sample(self.input_vocab, self.num_encoder_tokens, 
                sent, self.max_sentence_len, pad_token)
        prediction = self.model.predict(sent, batch_size=1)[0, 0]
        prediction = tf.argmax(prediction)

        prediction = idx2word(self.ref_classes, prediction)
        
        return [prediction] + [None for i in range(beam_search_width - 1)]

    def batch_predict(self, data, beam_search_width):
        results = []
        for el in data.iloc:
            results.append(self.predict(el["sentence"].text, beam_search_width))
        return results

    def eval(self, eval_df):
        return self.model.eval(eval_df)


