import re

from src.utils.regular_expressions import reference_pattern


class ReferenceMatcher(object):
    """
    Class for matching section-references (§) in spaCy.
    Further information: https://de.wikipedia.org/wiki/Zitieren_von_Rechtsnormen
    """

    name = "reference_matcher"
    expression = reference_pattern

    def __call__(self, doc):
        """
        Call method adds entities for document.
        Return:
        - doc: document with tagged entities
        """
        for match in re.finditer(self.expression, doc.text):

            start, end = match.span()
            span = doc.char_span(start, end, label="SECTION_REFERENCE")
            if (
                span is not None
                and match.group("book") is not None
                # Filter book
                # and "VwGO" in match.group("book")
            ):
                doc.ents = list(doc.ents) + [span]
        return doc
