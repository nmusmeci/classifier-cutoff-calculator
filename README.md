# Classifier profit calculator
Python library to calculate the expected net profit that will result from the application of a classifier in a commercial setting. The expected net profit is calculated for each value of the classifier's threshold (hence a net profit curve).

It can also be used to choose an optimal cut-off for the classifier that is problem-specific and has a clear business objective, as opposed to e.g. maximising F1 score.

It takes as input a profit-cost matrix that quantifies the commercial impact for each one of the 4 cases in the confusion matrix (true positive, true negative, false positive and false negative). This profit-cost matrix does not come from the data but needs to be derived from the domain experts of the specific application. 