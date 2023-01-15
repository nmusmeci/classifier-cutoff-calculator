# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:59:08 2023

@author: nicol
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'src/main/'))
from classifier_cutoff_calculator import ClassifierCutoffCalculator

def test_generate_net_gain_curve():
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_score = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    tp_gain = 2
    fp_cost = 1
    calculator = ClassifierCutoffCalculator(y_true, y_score, tp_gain, fp_cost)
    calculator.generate_net_gain_curve()
    expected_net_gain_values = [0, 0.25, 0.5, 0.25, 0.75, 0.5]
    expected_net_gain_index = [1.8, 0.8, 0.7, 0.5, 0.3, 0.1]

    assert calculator.expected_net_gain_series.tolist() == expected_net_gain_values
    assert calculator.expected_net_gain_series.index.tolist() == expected_net_gain_index
    
def test_find_optimal_cutoff():
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_score = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    tp_gain = 2
    fp_cost = 1
    calculator = ClassifierCutoffCalculator(y_true, y_score, tp_gain, fp_cost)
    calculator.generate_net_gain_curve().find_optimal_cutoff()

    assert calculator.optimal_cutoff == 0.3
    assert calculator.expected_net_gain_max == 0.75

    
if __name__ == '__main__':
    pytest.main()