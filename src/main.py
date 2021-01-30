from src.completion import Completion

"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""

if __name__ == "__main__":
    c = Completion()
    c.feed_data(key="test_ce_bverwg")
    c.train_data()
    c.evaluate()
