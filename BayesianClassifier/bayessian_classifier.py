import numpy
import scipy


""" The third lab will be devoted to the naive Bayesian classifier.
The dataset is already divided into 10 parts for cross-validation. The task is to classify spam.
Spam messages contain spmsg in their title, normal messages contain legit. 
The text of the letter itself consists of two parts: the subject and the body of the letter. 
All words are replaced by int corresponding to their index in some global dictionary (a kind of anonymization).
 Accordingly, you are required to build a naive Bayesian classifier and, in doing so,

1) Come up with, or test what you can do with the subject and body's letter to improve the quality of work.
2) How to take into account (or not to take into account) the words that may occur in the training sample, 
but may not meet in the test sample and vice versa.
3) How to impose additional restrictions on your classifier so that good letters almost never get into spam, 
but at the same time, perhaps the overall quality of the classification hasn't decreased too much.
4) Understand how the classifier is arranged inside and be able to answer any questions about the theory associated
with it.

For writing the classifier it is allowed to use numpy, scipy and pandas. Cross-validation can be done by any library."""


if __name__ == "__main__":
    pass