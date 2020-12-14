import unittest

import pandas as pd

from src.utils.stopwords import read_stopwords_csv


class StopwordlistTest(unittest.TestCase):
    def test_read_csv(self):
        """ Check stopword file content """
        df = read_stopwords_csv()
        self.assertEqual(type(pd.DataFrame()), type(df))
        self.assertEqual(
            set(df.columns.values),
            set(
                [
                    "Allgemein",
                    "Abkuerzungen",
                    "Zahl-roemisch",
                    "Zahl-woerter",
                    "Monate",
                    "Seitenzahlen",
                    "Buchstaben",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
