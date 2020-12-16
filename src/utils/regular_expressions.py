import re

"""
Regex pattern matching law references in german law
"""


reference_pattern = re.compile(
    r"(§|Art\.)[ \n]+(?P<section>[\d+|[ivxclmIVXCLM]+)+([ \n]+|[ \n]+\w|\w[ \n]+)*"
    r"(Abs\.[ \n]+\d+)*"
    r"([ \n]+UAbs\.[ \n]+\d+)*"
    r"([ \n]+Satz[ \n]+\d+|[ \n]+S\.[ \n]+\d+)*"
    r"([ \n]+Nr\.[ \n]+\d+)*"
    r"([ \n]+Spiegelstrich[ \n]+\d+)*"
    r"([ \n]+lit\.[ \n]+[a-z]+)*"
    r"([ \n]+sublit\.[ \n]+[ivxclmIVXCLM]+)*"
    r"([ \n]+Hs\.[ \n]+\d+)*"
    r"(?P<book>[ \n]+[A-ZÄÖÜß]+[a-zöäüß]*[A-ZÖÄÜß]+\w*)*",
    flags=re.MULTILINE,
)
