# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:38:59 2023

@author: nicolo
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from typing import List, Tuple

class ClassifierCutoffCalculator:
    def __init__(self, y_true: List[int], y_score: List[float], tp_gain: float,
                 fp_cost: float, tn_gain: float = 0., fn_cost: float = 0.,
                 p_1: float = None):
        """
        Initialize the ClassifierCutoffCalculator instance.
        
        Parameters
        ----------
        y_true : list or array of binary (1/0)
            Real value of the target variable for each data point in a test set.
            To avoid data likeage, this should not be the same as the training set.
        y_score : list or array of float, shape (,len(y_true))
            Score assigned by the classifier to each data point in a test set
            (higher scores mean higher confidence in target variable being 1).
            To avoid data likeage, this should not be the same as the training set.
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
        p_1: float, optional (default = None)
            Prior for class 1
        """
       
        self.y_true = y_true
        self.y_score = y_score
        self.tp_gain = tp_gain
        self.fp_cost = fp_cost
        self.tn_gain = tn_gain
        self.fn_cost = fn_cost
        self.p_1 = p_1
        self.expected_net_gain_series = None
        self.expected_net_gain_max = None
        self.optimal_cutoff = None
        
    def generate_net_gain_curve(self) -> None:
        """
        Calculate the net gain curve for the classifier.
        
        The net gain curve shows the expected net gain for each possible 
        cut-off of the classifier.
        The curve is calculated using a gain-cost matrix associated with each 
        element of the confusion matrix.
        
        This method saves the resulting net gain curve as a class attribute
        for later use.
        """
        
        fpr, tpr, cutoff = roc_curve(self.y_true, self.y_score)
        tnr = 1. - fpr
        fnr = 1. - tpr
        if self.p_1 is None:
            self.p_1 = np.mean(self.y_true)
        net_gain = self.p_1*(tpr*self.tp_gain - fnr*self.fn_cost) + \
            (1. - self.p_1)*(tnr*self.tn_gain - fpr*self.fp_cost)
        self.expected_net_gain_series = pd.Series(net_gain,index=cutoff)
        
        return self
        
    def find_optimal_cutoff(self) -> None:
        """
        Find the optimal cut-off for the classifier.
        
        The optimal cut-off is the value that maximizes the expected net gain 
        for the classifier.
        This method saves the maximum expected net gain and the optimal 
        cut-off as class attributes for later use.
        """
        
        self.expected_net_gain_max = self.expected_net_gain_series.max()
        self.optimal_cutoff = self.expected_net_gain_series.idxmax()
        
        return self
    
    def plot_net_gain_curve(self, figsize: Tuple[int, int] = (10,5)) -> None:
        """
        Plot the expected net gain curve for the classifier.
        
        The net gain curve shows the expected net gain for each possible 
        cut-off of the classifier.
        The curve is calculated using a gain-cost matrix associated with each 
        element of the confusion matrix.
        
        The optimal cut-off, as well as a horizontal line at y=0, are also 
        highlighted in the plot.
        
        Parameters
        ----------
        figsize: tuple, optional (default=(10,5))
            Width and height of the figure in inches.
            
        Raises
        ------
        ValueError
            If the net gain curve has not been calculated yet (using the 
            'generate_net_gain_curve' method).
        """
        
        if self.expected_net_gain_series is None:
            raise ValueError("The net gain curve has not been calculated yet."
                             "Please call the 'calculate_net_gain_curve' method first.")
        
        fig,ax = plt.subplots(1,1,figsize=figsize)
    
        self.expected_net_gain_series.plot(ax=ax,fontsize=12)
        ax.axhline(y=0, c='black', linestyle='--', linewidth=1)
        ax.axvline(x=self.optimal_cutoff, c='black', linestyle=':', linewidth=1)
    
        ax.set_title(f'Max net gain = {round(self.expected_net_gain_max,1)}',
                     fontsize=20)
        ax.set_xlim([0., 1.])
        ax.set_xlabel("Classifier's cut-off", fontsize=15)
        ax.set_ylabel("Expected net gain (per case)", fontsize=15)
        cutoff_annotation_offset = self.optimal_cutoff*0.01 \
            if self.optimal_cutoff < self.expected_net_gain_series.mean() \
            else -0.01*self.optimal_cutoff
        ax.annotate(f'Cut-off = {round(self.optimal_cutoff,2)}', 
                    (self.optimal_cutoff+cutoff_annotation_offset, 
                     self.expected_net_gain_max*0.1))
