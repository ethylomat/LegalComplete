# Possible task formulations:

### Inputs:
1. Sequence: fixed length or since sentence/document start
1. Sequence of word embeddings
1. Ngram (with fixed n):
1. Ngram of word embedding


### Optional Additional inputs
1. Document embedding (with doc2vec)
1. Previous books
    * Potentially only the last previous one relevant
1. Previous references
    * Potentially only the last previous one relevant
1. Document classes (Verfahrensart, Urteil/Beschluss, ..)
    * Downsides: 
    * Manual and inflexible
    * Requires labels which might not be known during inference time

### Outputs:
1. Sequence
    * Output is a sequence of words
    * Problems: how to obtain several options with probabilities?
1. Word
    * Sample several times to obtain entire reference prediction (with beam search)
1. Reference Class:
    * Categorical output: Every reference is a class 
    * Problems: a lot of classes.
1. three classes:
    * Outputs book, paragraph and section. For each output there is also a “blank”
    * Problem: more work setting up 3 classes

