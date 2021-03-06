import glob
import json
import os
from zipfile import ZipFile

import requests
from tqdm import tqdm

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

"""
Functions for handling datasets
"""


def get_available_datasets():
    """
    Listing available datasets (from .json files in datasets directory)

    Returns:
        datasets: List of dictionaries containing information about downloadable datasets
    """
    files = [file for file in glob.glob(os.path.join(MODULE_ROOT, "datasets/*.json"))]
    datasets = []
    for file in files:
        with open(file, "r") as f:
            dataset_info = json.load(f)
        datasets.append(dataset_info)
    return datasets


def get_dataset_info(key: str):
    """
    Returns dict of dataset info

    Args:
        key: String with dataset key (or filename)

    Returns:
        dataset_info: Dictionary with dataset information
    """
    key = key.lower().replace("-", "_").split(".")[0]
    filename = key + ".json"

    if filename not in os.listdir(os.path.join(MODULE_ROOT, "datasets")):
        raise FileNotFoundError

    with open(os.path.join(MODULE_ROOT, "datasets/", filename), "r") as f:
        dataset_info = json.load(f)
        return dataset_info


def download(url: str, filename: str):
    """
    Downloads file from url (with progress bar)

    Args:
        url: url of the source
        filename: filename of the file to write to

    Returns:
        success: Whether or not function downloaded file from source.
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    file = open(os.path.join(DATA_DIR, filename), "wb")
    bar = tqdm(
        desc=filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    )
    for data in resp.iter_content(chunk_size=1024):
        size = file.write(data)
        bar.update(size)
    file.close()
    return True


def download_dataset(
    dataset: [str, dict], force: bool = False, keep: bool = False
) -> bool:
    """
    Download datasets function

    Arguments:
        dataset: Dataset key or dataset_info dict
        force: Re-downloads file.
        keep: If not keep, .zip file is deleted

    Returns:
        success: Whether or not function downloaded file from source.
    """

    if type(dataset) == str:
        dataset = get_dataset_info(dataset)

    downloaded = False

    print()
    print("-" * 100)
    print("Downloading Dataset:")
    print("key:\t\t", dataset["key"])
    print("title:\t\t", dataset["title"])
    print("homepage:\t", dataset["homepage"])

    url = dataset["url"]
    filename = url.split("/")[-1]
    extracted = dataset["extracted"]

    if extracted not in os.listdir(DATA_DIR) or force:
        downloaded = download(url, filename)
    else:
        print("File already exists: ", extracted)

    if filename.endswith(".zip") and filename in os.listdir(DATA_DIR):
        extract_dataset(os.path.join(DATA_DIR, filename))
        if not keep:
            print("Removing zip file: ", filename)
            os.remove(os.path.join(DATA_DIR, filename))
    print("-" * 100)
    return downloaded


def download_all_datasets():
    """
    Download all available datasets

    Return:
        success: Whether or not function completed successfully.
    """
    print("Downloading all datasets ...")
    for dataset in get_available_datasets():
        download_dataset(dataset)


def extract_dataset(filename):
    """
    Extracts zip file of dataset.

    Args:
        filename: filename of dataset to be unzipped.
    """
    print("=" * 40)
    print("Extracting dataset: ", filename)
    zipfile = ZipFile(os.path.join(DATA_DIR, filename))
    zipfile.extractall(DATA_DIR)


if __name__ == "__main__":
    download_all_datasets()
