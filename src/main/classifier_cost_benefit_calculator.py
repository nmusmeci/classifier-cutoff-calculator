import pandas as pd
from sklearn.metrics import roc_curve

def net_benefit_curve(y_true, y_score, tp_gain, fp_cost, tn_gain=0., fn_cost=0.):

    """Calculate net benefit curve of a classifier for each possible threshold, given a profit-cost matrix
    associated with each element of the confusion matrix.
    
    Parameters
    ----------
    y_true : list or array of binary (1/0)
        Real value of the target variable for each data point 
    y_score : list or array of float, shape (,len(y_true))
        Score assigned by the classifier to each data point (higher scores mean higher confidence 
        in target variable being 1)
    tp_gain: float
        Profit or gain from correctly prediciting that a positive data point is positive.
        It must be > 0.
    fp_cost: float
        Loss from incorrectly predicting that a negative data point is positive.
        It must be > 0. 
    tn_gain: float, optional (default = 0.)
        Profit or gain from correctly prediciting that a negative data point is negative.
        It must be > 0.
    fn_cost: float, optional (default = 0.)
        Loss from incorrectly predicting that a positive data point is negative.
        It must be > 0. 
            
    Returns
    -------
    net_profit_series : Series, shape (,len(fpr))
        Series containing the net profit expected by using the classifier 
        as a function of the classifier's cut-off (Series' index is the proportion 
        of the total population classified as positive)
    optimal_threshold : float (range: 0-1)
        Proportion of the total population classified as positive that yields
        the highest net profit
    """

    # calculate total number of samples, total number of positive cases and
    # total number of negative cases
    size_population = y_true.shape[0]
    size_positives = sum(y_true==1)
    size_negatives = sum(y_true==0)
    
    # calculate false and true positive rates for each threshold in the classifier
    fpr,tpr, _ = roc_curve(y_true,y_score)
    # calculate false and true positive counts for each threshold from rates
    fp = fpr*(size_negatives)
    tp = tpr*(size_positives)
    # derive true and false negative counts from false and true positive counts,
    # to get the complete confusion matrix at each threshold
    tn = size_negatives - fp
    fn = size_positives - tp

    # use the complete confusion matrix to calculate net profit at each threshold  
    net_profit = tp*tp_gain + tn*tn_gain - fp*fp_cost - fn*fn_cost
    # calculate percentage of total population that is classified as positive for each
    # value of the threshold (to be used as index in the Pandas Series)
    tot_predicted_positives = (fp + tp)/size_population

    net_profit_series = pd.Series(net_profit,index=tot_predicted_positives)
    optimal_threshold = net_profit_series.idxmax()

    return [net_profit_series, optimal_threshold]