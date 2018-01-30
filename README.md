
# Part-of-speech tagger using Hidden Markov Model

Implemented part-of-speech tagger using bigram hidden markov model decoding - given an observation sequence of n words and a set of hidden states (POS tags) tn1, choose the most probable sequence of tags.

Derived the most probable tag sequence using Viterbi algorithm.

## Dataset

The dataset in English from Universal Dependencies (http://universaldependencies.org/#en) will be used for training and testing the model. Some of the pre-processed files used are described below:

 - __train.counts__: Frequency counts from the training file. It has two types of counts:
     - Emission counts, in the format __< count > WORDTAG < tag > < word >__
     - n-gram counts (where N is 1, 2 or 3), in the format __< count > < N-gram > < tag 1 > ... < tag N >__
     
 - __test.words__: Simplified version of the original test file, with one word per line and black lines separating sentence
 
 - __test.tags__: Simplified version of the original test file, with word and the true tag on each line and blank lines separating sentences __< word > < tag >__

### Supported python versions 

-  Python 3

## Documentation

Run the below command to store output tags derived using Viterbi algorithm for each word in test in a separate file in same format as test.tags(<word> <tags>). This will generate a predictions file.

 - python ViterbiAlgorithm.py < train.counts > < test.words > < predictions filename >

Use the eval_tagger.py to compute accuracy of the predictions against actual tags.

 - python eval_tagger.py < predictions filename > < test.tags >
