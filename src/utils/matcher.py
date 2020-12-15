import re


class ReferenceMatcher(object):
    """
    Class for matching section-references (§) in spaCy.
    """

    name = "reference_matcher"
    expression = r"(§|Art\.)[ \n]+(\d+)+([ \n]+|[ \n]+\w|\w[ \n]+)*(Abs\.[ \n]+\d+)*([ \n]+Satz[ \n]+\d+|[ \n]+S\.[ \n]+\d+)*([ \n]+Nr\.[ \n]+\d+)*([ \n]+Hs\.[ \n]+\d+)*(?P<book>[ \n]+[A-Z]+[a-z]*[A-Z]+\w*)*"

    def __call__(self, doc):
        """
        Call method adds entities for document.
        Return:
        - doc: document with tagged entities
        """
        for match in re.finditer(self.expression, doc.text, flags=re.MULTILINE):

            start, end = match.span()
            span = doc.char_span(start, end, label="SECTION_REFERENCE")
            if (
                span is not None
                and match.group("book") is not None
                and "VwGO" in match.group("book")
            ):
                doc.ents = list(doc.ents) + [span]
        return doc
