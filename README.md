# LegalComplete
![Tests](https://github.com/ethylomat/LegalComplete/workflows/Tests/badge.svg)
![Linting](https://github.com/ethylomat/LegalComplete/workflows/Linting/badge.svg)

Auto-reference for legal documents.

This projects aims to generate a autocompletion functionality for legal documents (judgements/decisions). 

The goal is a helpful tool for adding norm references. If the user inputs “Die auf Verfahrensmängel gemäß” or “Verfahrensmängel nach” the tool should recommend `§ 132 Abs. 2 Nr. 3 VwGO` as it may be the most often referenced norm for similar text input. 

## Prerequisites

First you need to clone the repository to receive the projects content:

```bash
$ git clone https://github.com/ethylomat/LegalComplete.git
Cloning into 'LegalComplete'...
```


It is recommended to install the Python packages local in a Pipenv environment:
```bash
$ cd LegalComplete
$ pipenv install --dev
```

For development make sure to have `pre-commit` installed in the project ([Documentation](https://pre-commit.com/#install)). 
```bash
$ pre-commit install
```
 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Used software

- Python Project Boilerplate
    - Source: [gh:jomazi/Python-Default](https://github.com/jomazi/python-default)
    - Parts used as project boilerplate (for project structure)
