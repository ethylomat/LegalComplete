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
 
## Project State

### Planning State

As of the moment we have finished our fist n-gram baseline approach (v1.0.0) for the recommendation of section (§)
references. The further steps will be the improvement considering the evaluation metrics (success rate) and 
speed of preprocessing the corpora.

Our first evaluation on the n-gram baseline approach shows the following success rates on the corpus of judgements
of the federal administrative court “Bundesverwaltungsgericht”:

```
Evaluation results (n-grams, n=4):                      
+----------------------+----------------+
| overall test samples | 4509           |
| correct (first)      | 1280 (0.28388) |
| correct (top 3)      | 1663 (0.36882) |
| incorrect            | 2846 (0.63118) |
| failed               | 0 (0.00000)    |
+----------------------+----------------+
```

The “correct (first)” rate is the count of correct suggestions, where the first suggestion was correct. The 
“correct (top 3)” rate is the count of suggestions, where the correct suggestion was in the top 3 of the suggestion
list.

For the future development higher success rates are intended for reliable recommendations (at least above 50% successful 
recommendations).

### Future Planning

To have higher success rates for the n-gram suggestions we will do further experiments to optimize the different parts.
Utilizing the baseline we have a measure to compare the effects of modifications. This allows us to further improve the
suggestions.

Another future task is the ability to ”trigger“ a suggestion. Suggestions could simply be triggered by typing a 
“§”-symbol but could also be triggered based on an n-gram approach. 

### High-level Architecture Description

At the moment we have a pipeline for the processing of datasets and further process them using custom objects for
different approaches.

The current n-gram baseline approach is mainly implemented in `src/completion_n_gram.py`. The n-gram completion class
defines the strategy of processing data and the actual n-gram based functionalities. The preprocessing is defined in 
`src/utils/preprocessing.py`. It uses the Python library [spaCy](https://github.com/explosion/spaCy) to create
(sentence, reference) pairs. For the detection of references we currently use a regular expression that is defined
in `src/utils/regular_expressions.py`. These regular expressions are used to annotate references as named entities
in spaCy which is done by the matcher in `src/utils/matcher.py`. For the preprocessing we use a pipeline of reference 
matcher and the spaCy sentencizer.

The generated sentence-reference pairs are further processed in `src/completion_n_gram.py`. Here the sentences are
normalized by lowercasing and removal of stopwords. In the future we will also test stemming. After processing the
n-grams are counted. This counts are used to calculate the prediction of the references.

The last step is the evaluation. Here we use a test set and simply test for the correct suggestions.

### Data Analysis

#### Data Sources

Like stated in the project proposal we currently use the open access datasets by 
[Seán Fobbe](https://zenodo.org/communities/sean-fobbe-data/).

For the first steps we will constrain to the corpus of jugements of the federal administrative court 
“Bundesverwaltungsgericht” [1]. Also we use his list of stopwords for legal texts [2].

In the future we will test on other corpora of german federal courts.  
List of datasets: [https://zenodo.org/communities/sean-fobbe-data/](https://zenodo.org/communities/sean-fobbe-data/) 

#### Preprocessing

#### Basic Statistics

#### Examples


### Experiments

## References 

[1] - Fobbe, Sean. (2020). Corpus der Entscheidungen des Bundesverwaltungsgerichts (CE-BVerwG) (Version 2020-06-23) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3911068  
[2] - Fobbe, Sean. (2020). Stoppwörter der Deutschen Rechtssprache (SW-DE-RS) (Version 1.0.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3995594  

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Used software

- Python Project Boilerplate
    - Source: [gh:jomazi/Python-Default](https://github.com/jomazi/python-default)
    - Parts used as project boilerplate (for project structure)
