import unittest

import pandas as pd

from src.completion import call_evaluate
from src.utils.preprocessing import build_pipeline


class TestPipeline(unittest.TestCase):
    def test_reference_entity_detection(self):
        """builds pipeline and checks whether a sample reference is annotated correctly"""
        text = (
            "Im Interesse der Verfahrensbeschleunigung macht der Senat von der"
            + "\nMöglichkeit der Zurückverweisung gemäß § 133 Abs. 6 VwGO Gebrauch."
        )

        nlp = build_pipeline(disable=["tagger", "parser", "ner"])
        doc = nlp(text)
        self.assertEqual(
            str(doc.ents[0]),
            "§ 133 Abs. 6 VwGO",
            "pipeline failed to output doc with reference entity anntoation",
        )
        self.assertEqual(
            doc.ents[0].label_,
            "SECTION_REFERENCE",
            "the label of a reference entity should be SECTION_REFERENCE",
        )


class TestEvaluation(unittest.TestCase):
    def test_evaluation(self):
        text = (
            "Im Interesse der Verfahrensbeschleunigung macht der Senat von der"
            + "\nMöglichkeit der Zurückverweisung gemäß § 133 Abs. 6 VwGO Gebrauch."
        )
        target = "§ 133 Abs. 6 VwGO"

        class MockedModel:
            def batch_evaluate(input_x, num_suggestions):
                results = ["" for i in range(num_suggestions)]
                results[0] = target
                print(results)
                return [results]

        data_test = pd.DataFrame({"text": [text]})
        refmodel = MockedModel
        nlp = build_pipeline(disable=["tagger", "parser", "ner"])
        metrics = call_evaluate(data_test, refmodel, nlp)
        print(metrics)
        self.assertEqual(metrics["first"], 1)
        self.assertEqual(metrics["three"], 1)
        self.assertEqual(metrics["incorrect"], 0)
