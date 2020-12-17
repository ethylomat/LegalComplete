import pandas as pd

from src.utils.common import read_csv, split_dataframe
from src.utils.retrieve import download_dataset, get_dataset_info

"""
Completion base class
"""


class Completion:
    data_train, data_test, data_dev = None, None, None

    def __init__(self):
        pass

    def feed_data(self, filename: str = "", key: str = ""):
        """
        Method for reading raw files into datasets
        Arguments:
        - filename: relative filename in data directory
        - key: alternative - give dataset key to receive automatically
        """

        # If key is provided instead of filename (-> filename is overwritten)
        if key:
            dataset_info = get_dataset_info(key)
            download_dataset(dataset_info)
            filename = dataset_info["extracted"]

        # TODO: Catch FileNotFoundError
        full_df = read_csv(filename)

        # Splitting dataframe into different sets
        self.data_train, self.data_test, self.data_dev = split_dataframe(
            full_df, fracs=[0.8, 0.003, 0.003]
        )

    def __str__(self):
        """
        String representation of completion objects.
        """
        class_name = str(self.__module__).split(".")[-1]
        return "<%s, Dataset: TRAIN %d | TEST %d | DEV %d>" % (
            class_name,
            self.data_train.shape[0],
            self.data_test.shape[0],
            self.data_dev.shape[0],
        )
