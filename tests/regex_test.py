import re
import unittest

from src.utils.regular_expressions import reference_pattern

"""
Test cases are based on examples of references in: https://de.wikipedia.org/wiki/Zitieren_von_Rechtsnormen
"""


class LawReferenceTest(unittest.TestCase):
    def test_simple_reference(self):
        """ Check simple reference is detected (section, paragraph)"""
        self.assertRegex("§ 1 BGB", reference_pattern)
        self.assertRegex("Art. 17 GG", reference_pattern)
        self.assertRegex("Art. 231 § 4 EGBGB", reference_pattern)
        self.assertRegex("Art. IX BWÜ", reference_pattern)

    def Test_consecutive_references(self):  # Enable by lowercasing method
        """ Check references with ranges are detected (e.g. §§ 12 ff. ZPO)"""
        self.assertRegex("§§ 12 f. ZPO", reference_pattern)
        self.assertRegex("§§ 12 ff. ZPO", reference_pattern)
        self.assertRegex("Art. 20 – 37 GG", reference_pattern)

    def Test_inserted_norm(self):  # Enable by lowercasing method
        self.assertRegex("Art. 20a GG", reference_pattern)
        self.assertRegex("§ 312f BGB", reference_pattern)


if __name__ == "__main__":
    unittest.main()
