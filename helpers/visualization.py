# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 11/04/2018
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_train_test_metric(train, test):
    """
    Create a plot showing the evolution of the same metric evaluated during epochs on train set & test set
    :param train: .csv filepath containing metric values evaluated on training data
    :param test: .csv filepath containing metric values evaluated on holdout data
    :return:
    """
    metric = np.loadtxt(train)
    val_metric = np.loadtxt(test)

    plt.plot(metric)
    plt.plot(val_metric)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

