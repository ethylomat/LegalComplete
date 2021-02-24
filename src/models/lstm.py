import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Input
from keras.models import Model

from src.utils.preprocessing import preprocess_fast

""" contains model with word2vec word embeddings as input and
Encoder-Decoder LSTM """


class LstmModel:
    def __init__(self, args, data_train):
        self.window_size = 5
        self.batch_size = 4
        self.epochs = 2
        self.max_sentence_len = 10
        self.max_decoder_seq_length = 5
        sentences = data_train["sentence"].text.to_list()
        sentences = [sentence.split(" ") for sentence in sentences]

        self.latent_dim = 50
        self.word_model = Word2Vec(
            sentences=sentences,
            vector_size=10,
            window=self.window_size,
            min_count=1,
            workers=4,
        )

        # a hacky way to get a onehot encoding for words from the target domain
        references = data_train["reference"].text.to_list()
        references = [reference.split(" ") for reference in references]
        target_word_model = Word2Vec(
            sentences=references, vector_size=1, window=1, min_count=1
        )
        self.target_vocab = target_word_model.wv

        print("finished setting up vocabulary")

        self.num_encoder_tokens = self.max_sentence_len
        self.num_decoder_tokens = self.max_sentence_len
        self.latent_dim = self.word_model.wv.vectors.shape[0]
        self.model = self.build_model()

    def build_model(self):
        """ builds and returns encoder decoder tensorflow model"""

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=encoder_states
        )
        decoder_dense = Dense(self.num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # inference setup
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        return model

    def _word2idx(self, word):
        return self.word_model.wv.key_to_index[word]

    def _idx2word(self, idx):
        return self.word_model.wv.index_to_key[idx]

    def _word2onehot(self, word):
        vect = np.zeros(self.num_encoder_tokens, dtype=np.uint8)
        vect[self._word2idx(word)] = 1
        return vect

    def target_word2idx(self, word):
        return self.target_vocab.key_to_index[word]

    def target_idx2word(self, idx):
        return self.target_vocab.index_to_key[idx]

    def preprocess(self, data):
        data = data.filter(["sentence", "reference"])
        for d in data.iloc:
            print(d["sentence"], " is d")

        return {"encoder_input": None, "decoder_input": None, "decoder_target": None}

    def convert_to_fixed_length_input(self, data):
        pass

    def train(self, train_df, eval_df):
        train_data = self.preprocess(train_df)
        # TODO: eval_df = self.preprocess(eval_df)

        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        self.model.fit(
            [train_data["encoder_input"], train_data["decoder_input"]],
            train_data["decoder_target"],
            batch_size=self.batch_size,
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
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_word2idx["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value
            )

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.target_idx2word[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (
                sampled_char == "\n"
                or len(decoded_sentence) > self.max_decoder_seq_length
            ):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]

        return decoded_sentence
