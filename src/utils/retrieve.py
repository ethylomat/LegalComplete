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
    Arguments:
    Return:
    - datasets: List of dictionaries containing information about downloadable datasets
    """
    files = [file for file in glob.glob(os.path.join(MODULE_ROOT, "datasets/*.json"))]
    datasets = []
    for file in files:
        with open(file, "r") as f:
            dataset_info = json.load(f)
    datasets.append(dataset_info)
    return datasets


def download(url: str, filename: str):
    """
    Downloads file from url (with progress bar)
    Arguments:
    - url: url of the source
    - filename: filename of the file to write to
    Return:
    - success: Whether or not function downloaded file from source.
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


def download_dataset(dataset_info, force=False, keep=False) -> bool:
    """
    Download datasets function
    Arguments:
    Return:
    - success: Whether or not function downloaded file from source.
    """
    downloaded = False

    print("=" * 120)
    print("Downloading Dataset:")
    print("key:\t\t", dataset_info["key"])
    print("title:\t\t", dataset_info["title"])
    print("homepage:\t", dataset_info["homepage"])

    url = dataset_info["url"]
    filename = url.split("/")[-1]
    extracted = dataset_info["extracted"]

    if extracted not in os.listdir(DATA_DIR) or force:
        downloaded = download(url, filename)
    else:
        print("File already exists: ", extracted)

    if filename.endswith(".zip") and filename in os.listdir(DATA_DIR):
        extract_dataset(os.path.join(DATA_DIR, filename))
        if not keep:
            print("Removing zip file: ", filename)
            os.remove(os.path.join(DATA_DIR, filename))
    return downloaded


def download_all_datasets():
    """
    Download all available datasets
    Arguments:
    Return:
    - success: Whether or not function completed successfully.
    """
    print("Downloading all datasets ...")
    for dataset in get_available_datasets():
        download_dataset(dataset)


def extract_dataset(filename) -> bool:
    print("=" * 40)
    print("Extracting dataset: ", filename)
    zip = ZipFile(os.path.join(DATA_DIR, filename))
    zip.extractall(DATA_DIR)
    return True


if __name__ == "__main__":
    download_all_datasets()
