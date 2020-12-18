## Pattern search
Norm references are not the only frequently occurring pattern in our dataset. In order to gain further insight on which patterns could be leveraged for a general text completion we searched the dataset for more such patterns. 

### Approach

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

### Results

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






### Observations

Norm references seem to appear more often than other patterns
Other frequent patterns are citations of past cases (usually BVerwG). These are related in structure to norm references, however when analysing the actual texts, it seems they often do not have easily identifiable flags (like the § used in norm references). Nonetheless this pattern could be leveraged for autocompletion.
Some very long segments with over 100 words are likely to have been copy pasted. Integrating such segments into autocompletion may be more convenient than autocompletion.
