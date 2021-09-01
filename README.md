# Classifier cost-benefit calculator
Python library to calculate the expected cost-benefit balance that will result from the application of a classifier in a commercial setting. 

The idea is that there is a cost incurred when the classifier is wrong and a benefit (or profit) when the classifier is right. Given these costs and benefits, the library calculates the expected cost-benefit balance for each value of the classifier's threshold. It also calculates the threshold that maximizes the net benefit (i.e. total benefit - total cost), providing a problem-specific and business-driven solution to the question of finding the optimal cut-off for a classifier (as opposed to generic approaches such as maximizing F1 score).

Note: numerical values of costs and benefits do not come from the data but need to be derived from the domain experts of the specific application. 
