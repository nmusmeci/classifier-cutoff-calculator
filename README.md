# Classifier cost-benefit calculator
Python package to calculate the expected <b>net benefit balance</b> that will result from the application of a classifier in a commercial setting. 

The idea is that there is a cost incurred when the classifier is wrong and a benefit (or profit) when the classifier is right. For example, a classifier that identifies customers to include in a marketing email will generate a profit when it recommends a customer who ends up converting, and a cost when it recommends a customer who doesn't convert and ends up unsubscribing from future marketing emails.

Given these costs and benefits, the package lets the user calculate the expected net benefit balance (i.e. total benefit - total cost) for each value of the classifier's threshold. It also calculates the threshold that maximizes the net benefit, providing a <b>problem-specific and business-driven solution to the question of finding the optimal cut-off</b> for a classifier (as opposed to generic approaches such as maximizing F1 score).

Note: numerical values of costs and benefits are inputs; they need to be derived from the specifics of the commercial application (e.g. in the example of the marketing email they could come from customer life-time models).

This approach is inspired by the chapter 7 of the book "[Data Science for Business](https://learning.oreilly.com/library/view/data-science-for/9781449374273/)" by Foster Provost and Tom Fawcett.
