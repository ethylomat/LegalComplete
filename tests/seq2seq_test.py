import unittest

import pandas as pd

from src.models.transformers_seq2seq import TransSeqModel
from src.models.word2vec import Word2VecModel


class Seq2SeqText(unittest.TestCase):
    def test_training(self):
        mock_data = [
            "Im Interesse der Verfahrensbeschleunigung macht der Senat "
            + "unerheblichkeit gemäß § 1 Abs. 2 VwGO oder vielleicht"
            + " unsachlichkeit gemäß § 2 Abs. 2 VwGO"
        ]

        mock_data.append(
            "Im Interesse der Verfahrensbeschleunigung macht der Senat von der"
            + "Möglichkeit der Zurückverweisung gemäß § 133 Abs. 6 VwGO Gebrauch."
        )
        mock_data = [mock_data[1] for i in range(20)]
        # dev_data = ["ob unerheblichkeit gemäß"]
        data_train = pd.DataFrame({"text": mock_data})
        predictor = TransSeqModel()
        predictor.train(data_train, data_train.copy())
