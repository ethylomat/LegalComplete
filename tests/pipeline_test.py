import unittest

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
