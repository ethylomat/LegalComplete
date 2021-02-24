import unittest

import pandas as pd

from src.completion import evaluate_references
from src.utils.preprocessing import build_pipeline, preprocess_fast


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

    def test_preprocess_fast(self):
        data = preprocess_fast(self.df_data)
        self.assertEqual(data.iloc[0]["reference"], self.target_reference)


class TestEvaluation(unittest.TestCase):
    def test_evaluation(self):
        text = (
            "Im Interesse der Verfahrensbeschleunigung macht der Senat von der"
            + "\nMöglichkeit der Zurückverweisung gemäß § 133 Abs. 6 VwGO Gebrauch."
        )
        target = "§ 133 Abs. 6 VwGO"

        class MockedModel:
            def batch_predict(input_x, num_suggestions):
                results = ["" for i in range(num_suggestions)]
                results[0] = target
                return [results]

        data_test = pd.DataFrame({"text": [text]})
        refmodel = MockedModel
        nlp = build_pipeline(disable=["tagger", "parser", "ner"])
        metrics = evaluate_references(data_test, refmodel, nlp)
        self.assertEqual(metrics["first"], 1)
        self.assertEqual(metrics["three"], 1)
        self.assertEqual(metrics["incorrect"], 0)
