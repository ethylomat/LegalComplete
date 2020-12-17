from src.completion_n_gram import NGramCompletion

"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""

if __name__ == "__main__":
    ngc = NGramCompletion()
    ngc.feed_data(key="ce-bverwg")
