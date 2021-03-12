import re

from spacy.language import Language

from src.utils.regular_expressions import reference_pattern


def match_reference(doc):
    expression = reference_pattern

    """
    Class for matching section-references (§) in spaCy.
    Further information: https://de.wikipedia.org/wiki/Zitieren_von_Rechtsnormen
    Call method adds entities for document.
    Return:
    - doc: document with tagged entities
    """
    for match in re.finditer(expression, doc.text):

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
