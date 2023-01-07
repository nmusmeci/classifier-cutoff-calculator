# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:52:59 2023

@author: nicolo
"""

from classifier_cutoff_calculator import ClassifierCutoffCalculator

# create an instance of the ClassifierCutoffCalculator class
ccc = ClassifierCutoffCalculator(y_true=[1, 1, 0, 1, 0, 0, 1, 0],
                                 y_score=[0.9, 0.7, 0.6, 0.75, 0.5,
                                          0.3, 0.8, 0.2],
                                 tp_gain=10, fp_cost=5)

# calculate the net gain curve
ccc.generate_net_gain_curve()

# calculate optimal cut-off
ccc.find_optimal_cutoff()

# plot the net gain curve
ccc.plot_net_gain_curve()