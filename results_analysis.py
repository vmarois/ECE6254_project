# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 03/04/2018
"""
from helpers import *

if __name__ == '__main__':
    #plot_train_test_metric('output/metrics_evolution/dnn_model_loss.csv', 'output/metrics_evolution/dnn_model_val_loss.csv')

    plot_sample(model='dnn', datapath='data/', phase='ES')
