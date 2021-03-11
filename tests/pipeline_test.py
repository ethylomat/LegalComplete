import unittest

import pandas as pd

from src.completion import Completion
from src.utils.preprocessing import build_pipeline  # , preprocess_fast


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.target_reference = "§ 133 Abs. 6 VwGO"
        self.text = (
            "Im Interesse der Verfahrensbeschleunigung macht der Senat von der"
            + "\nMöglichkeit der Zurückverweisung gemäß § 133 Abs. 6 VwGO Gebrauch."
        )
        self.df_data = pd.DataFrame({"text": [self.text]})

    def test_reference_entity_detection(self):
        """builds pipeline and checks whether a sample reference is annotated correctly"""
        nlp = build_pipeline(disable=["tagger", "parser", "ner"])
        doc = nlp(self.text)
        self.assertEqual(
            str(doc.ents[0]),
            self.target_reference,
            "pipeline failed to output doc with reference entity anntoation",
        )
        self.assertEqual(
            doc.ents[0].label_,
            "SECTION_REFERENCE",
            "the label of a reference entity should be SECTION_REFERENCE",
        )
