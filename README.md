# LegalComplete

Auto-reference for legal documents.

This projects aims to generate a autocompletion functionality for legal documents (judgements/decisions). 

The goal is a helpful tool for adding norm references. If the user inputs “Die auf Verfahrensmängel gemäß” or “Verfahrensmängel nach” the tool should recommend `§ 132 Abs. 2 Nr. 3 VwGO` as it may be the most often referenced norm for similar text input. 

## Prerequisites

First you need to clone the repository to receive the projects content:

```bash
$ git clone https://github.com/ethylomat/LegalComplete.git
Cloning into 'LegalComplete'...
```


It is recommended to install the Python packages local in a virtual environment. Therefore you need `python3`, `python3-pip` and `python3-venv` installed on your system. You can create a virtual environment as follows:
```bash
$ cd LegalComplete
$ python3 -m venv venv
$ . venv/bin/activate # <- the “.” is important
(venv) $  # <-- activated virtual environment
```

Install the required python package with pip:

```bash
$ pip install -r requirements.txt
 ```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Used software

- Python Project Boilerplate
    - Source: [gh:jomazi/Python-Default](https://github.com/jomazi/python-default)
    - Parts used as project boilerplate (for project structure)
