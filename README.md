# Classifier cost-benefit calculator
Python functions that calculate the expected <b>net benefit balance</b> that will result from the application of a classifier in a commercial setting. 

Whenever a classifier is used in a commercial setting there is a cost incurred when the classifier is wrong and a benefit (or profit) when the classifier is right. For example, a classifier that identifies customers to include in a marketing email will generate a profit when it recommends a customer who ends up converting, and a cost when it recommends a customer who doesn't convert and actually ends up unsubscribing from future marketing emails.

The functions in this repo let the user calculate the expected cost-benefit balance (i.e. total benefit - total cost) for each value of the classifier's threshold. As inputs, the functions need a quantitative estimate of these costs and benefits, as well as data about the classifier out-of-sample performance (y_true and y_score). They also calculate the threshold that maximizes the net benefit, providing a <b>problem-specific and business-driven solution to the question of finding the optimal cut-off</b> for a classifier (as opposed to generic approaches such as maximizing F1 score).

Note: numerical values of costs and benefits are inputs; they need to be derived from the specifics of the commercial application (e.g. in the example of the marketing email they could come from customer life-time value models).

This approach is inspired by the chapter 7 of the book "[Data Science for Business](https://learning.oreilly.com/library/view/data-science-for/9781449374273/)" by Foster Provost and Tom Fawcett.
