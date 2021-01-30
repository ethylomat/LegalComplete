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
| overall test samples | 1002 (1.00000) |
| correct (first)      |  397 (0.39621) |
| correct (top 3)      |  473 (0.47206) |
| incorrect            |  529 (0.52794) |
| failed               |    0 (0.00000) |
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
| First     | [VwGO 99 2, 1606]     |
| Second    | [VwGO 99, 862]        |
| Third     | [VwGO 100, 644]       |
| Fourth    | [VwGO 154 2, 184]     |
| Fifth     | [VwGO 99 1, 154]      |
+-----------+-----------------------+
```

Additionally the mean and variance of occurrences per document per subgroup shows how many references we can expect per document and how (in-)consistent this user behaviour probably is (files: mean_ref_occurrence_per_doc(/_variance)).

As for all three categories the general mean reference occurrence is in the range of 5 to 9, whereas the variance is somewhere between 40 to 90, thus this is probably not a feasible and meaningful parameter to safely consider.
A certain minimum of reference mentioning can be expected, but e.g. the co-occurrences should probably not be chained to the total amount of occurrences per document, especially as the amount is not foreseeable.


### Pattern search
Norm references are not the only frequently occurring pattern in our dataset. In order to gain further insight on which patterns could be leveraged for a general text completion we searched the dataset for more such patterns. 

#### Approach

The main principle is to search for repetitions in text in an automated fashion. Those repetitions with a high frequency are relevant. Additionally the longer a repeating phrase is, the more valuable it is for use in auto completion.

One way of finding repetitions is to set up ngrams of varying lengths and count their frequencies. This can become inefficient for large tests. Suffix trees (or suffix arrays) can be built in  O(n), thus are more suited than ngrams, however in our initial attempts there was no available library we could easily adapt for our pipeline, thus we used NLTKs everygrams instead.

The only preprocessing used here was lower case conversion. Stemming was not used because the grammar that would be necessary to reproduce the exact formulation was assumed to be too complex for our autocompletion. 

Note that if a phrase with 30 words appears 50 times, then there are several shorter n grams which are variations of the same phrase that also appear 50 times. Therefore filtering was performed, such that a subphrase was only included if it appeared significantly more often than its parent phrase (2x more often).

#### Named Entity Removal

In the case of norm references a pattern consists of named entities such as section numbers or the respective statute books (Gesetzbücher). This analysis is not concerned with prediction of individual patterns, but the detection of patterns. Therefore we can replace all statute books and section numbers with a symbol, in order to capture such a pattern in a repetition search. 

Example: “§ 132 abs. 2” becomes “§ <number> abs. <number>”
In a search of 5% of the entire dataset this pattern appears 13942 times.

Spacy is capable of detecting many different entities such as names. However replacing those can cause issues. Buchholz is a name of a book that is very frequently referenced (this is different to a norm reference as this is not a statute book), but there are others as well. These book names should be handled differently than names of people involved in the cases. Hence in the initial experiments we only used the following symbols:
Dates (and Months)
Numbers
Statute books

#### Results

The figures show a dot per frequent ngram. Only the top ngram per ngram length is shown.

![alt text](https://github.com/ethylomat/LegalComplete/blob/main/doc/img/pattern_search/frequent_phrases_ngrams_length_4-14.png)

Frequencies of longer ngrams (from 15-50) can be seen in the second figure.

![alt text](https://github.com/ethylomat/LegalComplete/blob/main/doc/img/pattern_search/frequent_phrases_ngrams_length_15-50.png)

The following is a list of the most frequently occurring pattern per n for ngrams with length n.
In this case only ngrams from length 4 to 14 are shown.

freq:  643 length 14
hat der <number>. senat des bundesverwaltungsgerichts am <datum> durch den vorsitzenden richter am bundesverwaltungsgericht

freq:  400 length 13
in der verwaltungsstreitsache -<number>- hat der <number>. senat des bundesverwaltungsgerichts am <datum> durch

freq:  366 length 12
in der verwaltungsstreitsache hat der <number>. senat des bundesverwaltungsgerichts am <datum> durch

freq:  510 length 11
vom <datum> - bverwg <number> <number>.<number> - buchholz <number> § <number>

freq:  757 length 10
vom <datum> - bverwg <number> <number>.<number> - bverwge <number>, <number>

freq:  438 length 9
vom <datum> - bverwg <number> <number>.<number> - buchholz <number>.<number>

freq:  1036 length 8
vom <datum> - bverwg <number> <number>.<number> - buchholz

freq:  2734 length 7
vom <datum> - bverwg <number> <number>.<number> -

freq:  4836 length 6
§ <number> abs. <number> satz <number>

freq:  2496 length 5
des § <number> abs. <number>

freq:  13942 length 4
§ <number> abs. <number>


#### Additional examples from longer ngrams:

freq:  22 length 35
bundesverwaltungsgericht, simsonplatz <number>, <number> leipzig, schriftlich oder in elektronischer form (verordnung vom <datum>, <gesetzbuch>l s. <number>) einzureichen. für die beteiligten besteht vertretungszwang; dies gilt auch für die begründung der revision. die beteiligten müssen sich durch

freq:  25 length 34
befähigung zum richteramt der zuständigen aufsichtsbehörde oder des jeweiligen kommunalen spitzenverbandes des landes, dem sie als mitglied zugehören, vertreten lassen. in derselben weise muss sich jeder beteiligte vertreten lassen, soweit er einen antrag stellt.

freq:  35 length 33
vom <datum> mit schriftsatz vom <datum> zurückgenommen. das beschwerdeverfahren ist deshalb in entsprechender anwendung von § <number> satz <number>, § <number> abs. <number> satz <number>, § <number> abs. <number> satz <number> <gesetzbuch> einzustellen.






#### Observations

Norm references seem to appear more often than other patterns
Other frequent patterns are citations of past cases (usually BVerwG). These are related in structure to norm references, however when analysing the actual texts, it seems they often do not have easily identifiable flags (like the § used in norm references). Nonetheless this pattern could be leveraged for autocompletion.
Some very long segments with over 100 words are likely to have been copy pasted. Integrating such segments into autocompletion may be more convenient than autocompletion.



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
