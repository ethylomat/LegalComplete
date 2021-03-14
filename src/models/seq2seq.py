import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model

from src.utils.preprocessing import preprocess


class Seq2seqModel:
    def __init__(self, args, data_train):
        self.window_size = 5
        self.batch_size = 4
        self.epochs = 10
        self.max_sentence_len = 10
        sentences = data_train["text"].to_list()
        sentences = [sentence.replace("\n", "").split(" ") for sentence in sentences]

        self.word_model = Word2Vec(
            sentences=sentences,
            vector_size=10,
            window=self.window_size,
            min_count=1,
            workers=4,
        )
        print("finished setting up vocabulary")

        self.num_encoder_tokens = self.max_sentence_len
        self.latent_dim = self.word_model.wv.vectors.shape[0]
        self.model = self.build_model()

    def build_model(self):
        encoder_inputs = Input(shape=(None,), dtype=tf.int32)
        x = Embedding(self.num_encoder_tokens, self.latent_dim)(encoder_inputs)
        x, state_h, state_c = LSTM(self.latent_dim, return_state=True)(x)
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        x = Embedding(self.num_encoder_tokens, self.latent_dim)(decoder_inputs)
        x = LSTM(self.latent_dim, return_sequences=True)(
            x, initial_state=encoder_states
        )
        decoder_outputs = Dense(self.num_encoder_tokens, activation="softmax")(x)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile & run training
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        # Note that `decoder_target_data` needs to be one-hot encoded,
        # rather than sequences of integers like `decoder_input_data`!
        return model

    def preprocess_seq2seq(self, data_train):
        data_train = preprocess(data_train)
        encoder_input = tf.sparse.SparseTensor(
            indices=tf.zeros(0, 3),
            values=tf.zeros([], dtype=tf.int32),
            dense_shape=(
                len(data_train),
                self.max_sentence_len,
                self.num_encoder_tokens,
            ),
        )

        decoder_input = tf.sparse.SparseTensor(
            indices=tf.zeros(0, 3),
            values=[],
            dense_shape=(
                len(data_train),
                self.max_sentence_len,
                self.num_encoder_tokens,
            ),
        )

        decoder_target = tf.sparse.SparseTensor(
            indices=tf.zeros(0, 3),
            values=[],
            dense_shape=(
                len(data_train),
                self.max_sentence_len - 1,
                self.num_encoder_tokens,
            ),
        )
        index = 0
        for sample in data_train.iloc:
            x = [self._word2onehot(word) for word in str(sample["sentence"]).split(" ")]
            label = [
                self._word2onehot(word) for word in str(sample["sentence"]).split(" ")
            ]

            encoder_input[index] = np.array(x[: self.max_sentence_len], dtype=np.int32)
            decoder_input[index] = np.array(
                label[: self.max_sentence_len], dtype=np.int32
            )
            decoder_target[index] = np.array(
                label[: self.max_sentence_len - 1], dtype=np.int32
            )
            index += 1
        encoder_input = np.array(encoder_input)

        decoder_input = np.array(decoder_input)

        decoder_target = np.array(decoder_target)

        return encoder_input, decoder_input, decoder_target

    def _word2idx(self, word):
        return self.word_model.wv.key_to_index[word]

    def _idx2word(self, idx):
        return self.word_model.wv.index_to_key[idx]

    def _word2onehot(self, word):
        vect = np.zeros(self.num_encoder_tokens, dtype=np.uint8)
        vect[self._word2idx(word)] = 1
        return vect

    def train(self, train_data, data_dev):
        encoder_input, decoder_input, decoder_target = self.preprocess_seq2seq(
            train_data
        )
        self.model.fit(
            [encoder_input, decoder_input],
            decoder_target,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
        )
