import os

from utils.retrieve import download_all_datasets

"""
This script is the entrypoint of the program.
Every module or package it relies on has to be imported at the beginning.
"""

if __name__ == "__main__":
    download_all_datasets()
