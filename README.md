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

#### Preprocessing:
Before any further processing took place, the data-set was tokenized and the sentences were filtered in such way, that only sentences with an occurring "§" were used.
This was done to reduce the size of the data-set and reduce computation times.


#### Basic Statistics:
The plots and csv. files can be accessed via: doc/img within the repository.
They are divided after subgroups with prefixes Total, Beschluss and Urteil.

A general statistical analysis was conducted for the complete data-set. 
For a more sophisticated analysis, this was repeated for different subgroups of decisions within the data-set.

##### General Analysis:

The general analysis engaged some questions of data-set overview, content and estimation of user behaviour.

First, the distribution of sentences and legal ruling was examined (file: general_distribution_of_procedure).
The plot shows how many sentences and legal rulings are in the data-set, additionally their sub-parts after type of procedure are displayed in the form (type of decision, type of procedure).
(Types of procedures: e.g. A: erstinstanzliche Klage, C: Revisionen in Verwaltungsstreitverfahren, ... )

The general distribution of the procedures shows us the main constituents of the data-set. 
By further subdividing the data-set, variances in relative occurrences and finally variances within the co-occurrence matrices for each subset might indicate the need to introduce criteria for increasing the suggestion rate per subdivision (if the subdivision is known or can be extracted).  


##### Subset Analysis:

The goal of the subset analysis was to determine if a detection of subgroups is necessary and / or feasible and if such a detection might (if introduced into the detection algorithm) improve the suggestion quality.

For all documents within the data-set (# 24224) and the major subgroups, which were chosen after the type of decision (Urteil (sentence), # 4131 ; Beschluss (legal ruling), # 20076), the most occurring references (with and without paragraph subdivision (files: references_all, Total_references_all_reduced)) and the occurring law-books mentioned were extracted (files: books_all, books_all_reduced). 

What catches the eye is, that the "Beschluss" subset is less diverse than the "Urteil" subset, in all "VwGO" is obviously most frequently mentioned.
The second most mentioned law-book is already different (Urteil: SGB, Beschluss: ZPO).
This is also observable by examining the reference plots.

##### Co-occurrences:

Furthermore, the co-occurrence matrices for occurring references were calculated and a top 5 of most co-occurrences per reference were saved as a .csv.
Comparing the subgroups, the dimensionality is reduced considerably by only interacting with certain subgroups (Total: # 4000, Urteil: # 2600, Beschluss: # 2500).
This however probably only results from rare references not being included.

By observing the changes within the more frequently used references, the order of co-occurrences and co-occurrence frequency is not stable.
E.g. (drastic example) VwGO 100 appears almost exclusively within the "Beschluss" subset, VwGO 100 occurrence to VwGO 99 2 appears 1600 times within "Beschluss", but never in "Urteil". 

Creating subset co-occurrence matrices for medium sized subdivisions of the data-set could therefore probably improve reference suggestion for detectable parameters (which identify subdivisions) and should be considered for further examination. 

```
Top 5 co-occurrences of VwGO 100:
+-----------+-----------------------+
| First 	| [VwGO 99 2, 1606]		|
| Second   	| [VwGO 99, 862]		|
| Third     | [VwGO 100, 644]		|
| Fourth    | [VwGO 154 2, 184]		|
| Fifth     | [VwGO 99 1, 154]		|
+-----------+-----------------------+
```

Additionally the mean and variance of occurrences per document per subgroup shows how many references we can expect per document and how (in-)consistent this user behaviour probably is (files: mean_ref_occurrence_per_doc(/_variance)).

As for all three categories the mean is in the range of 5 to 9, whereas the variance is somewhere between 40 to 90, thus this is probably not a feasible and meaningful parameter to safely consider.
A certain minimum of reference mentioning can be expected, but e.g. the co-occurrences should probably not be chained to the total amount of occurrences per document, especially as the amount is not foreseeable.


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
