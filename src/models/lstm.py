import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model

from src.utils.preprocessing import (
    idx2word,
    preprocess_fast,
    process_input_sample,
    seq2seq_generator,
    word2idx,
    word2onehot,
)

""" contains model with word2vec word embeddings as input and
Encoder-Decoder LSTM """


class LstmModel:
    def __init__(self, args, data_train, input_vocab, target_vocab):
        self.target_vocab = target_vocab
        self.input_vocab = input_vocab
        self.batch_size = 1
        self.epochs = 2
        self.max_sentence_len = 7
        self.max_decoder_seq_length = 7

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
        self.num_decoder_tokens = self.target_vocab.vectors.shape[0]
        print("finished init setup")
        self.model = self.build_model()
        print("finished building model")

    def build_model(self):
        """ builds and returns encoder decoder tensorflow model"""

        encoder_inputs = Input(shape=(None,), name="encoder_input")
        x = Embedding(self.num_encoder_tokens, self.latent_dim)(encoder_inputs)
        x, state_h, state_c = LSTM(self.latent_dim, return_state=True)(x)
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name="decoder_input")
        x = Embedding(
            self.num_decoder_tokens, self.latent_dim, input_length=self.max_sentence_len
        )(decoder_inputs)

        decoder_lstm = LSTM(self.latent_dim, return_sequences=False, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())

        # inference setup
        decoder_lstm = LSTM(self.latent_dim, return_sequences=False, return_state=True)

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,), name="decoder_state_h")
        decoder_state_input_c = Input(shape=(self.latent_dim,), name="decoder_state_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            x, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            {
                "decoder_inputs": decoder_inputs,
                "decode_state_h": decoder_state_input_h,
                "decoder_state_c": decoder_state_input_c,
            },
            [decoder_outputs] + decoder_states,
        )

        return model

    def preprocess(self, data):
        return seq2seq_generator(
            data,
            self.target_vocab,
            self.num_decoder_tokens,
            self.input_vocab,
            self.num_encoder_tokens,
            self.max_sentence_len,
            self.max_decoder_seq_length,
        )

    def convert_to_fixed_length_input(self, data):
        pass

    def train(self, train_df, eval_df):
        train_data = self.preprocess(train_df)
        eval_data = self.preprocess(eval_df)

        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        self.model.fit(
            train_data,
            steps_per_epoch=5,
            validation_data=eval_data,
            epochs=self.epochs,
        )

    def predict(self, sent, beam_search_width):
        # TODO: implement multiple suggestions with beam_search_width param
        # for now only the first suggestion is made, others are set to None
        prediction = self.decode_sequence(sent)
        return [prediction] + [None for i in range(beam_search_width - 1)]

    def batch_predict(self, data, beam_search_width):
        results = []
        for el in data.iloc:
            results.append(self.predict(el["sentence"].text, beam_search_width))
            print(el["sentence"], "->", results[-1])
        return results

    def eval(self, eval_df):
        return self.model.eval(eval_df)

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        start_token = word2onehot(self.input_vocab, "<start>", self.num_encoder_tokens)
        input_vector = process_input_sample(
            self.input_vocab,
            self.num_encoder_tokens,
            input_seq,
            self.max_sentence_len,
            start_token,
        )
        print("in vect: ", input_vector.shape)
        state_c, state_h = self.encoder_model.predict([input_vector], batch_size=1)
        """
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, word2idx(self.target_vocab, "§")] = 1.0
        """
        decoder_input = word2onehot(
            self.target_vocab, "<start>", self.num_decoder_tokens
        )
        decoder_input = np.reshape(decoder_input, (1, 1, self.num_decoder_tokens))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        print(self.encoder_model.summary())
        state_c = np.reshape(state_c, (1, self.max_decoder_seq_length, self.latent_dim))
        state_h = np.reshape(state_h, (1, self.max_decoder_seq_length, self.latent_dim))
        x = {
            "decoder_inputs": decoder_input,
            "decode_state_h": state_h,
            "decoder_state_c": state_c,
        }

        print(self.decoder_model.summary())
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(x, batch_size=1)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = idx2word(self.target_vocab, sampled_token_index)
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (
                sampled_char == "<end>"
                or len(decoded_sentence) > self.max_decoder_seq_length
            ):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            state_c, state_h = c, h

        return decoded_sentence
