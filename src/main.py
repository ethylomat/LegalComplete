from src.utils.preprocessing import preprocess

"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""

if __name__ == "__main__":
    preprocess("example_dataset.csv")
