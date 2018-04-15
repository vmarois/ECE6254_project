# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 03/04/2018
"""
from helpers import *

if __name__ == '__main__':
    """
    # plot loss evolution of DNN model
    plot_train_test_metric('output/metrics_evolution/dnn_model_loss.csv', 'output/metrics_evolution/dnn_model_val_loss.csv', 'DNN Loss Evolution', 'Loss')

    # plot accuracy evolution of DNN model
    plot_train_test_metric('output/metrics_evolution/dnn_model_acc.csv',
                           'output/metrics_evolution/dnn_model_val_acc.csv', 'DNN Accuracy Evolution', 'Accuracy')

    # plot loss evolution of CNN model
    plot_train_test_metric('output/metrics_evolution/cnn_model_loss.csv',
                           'output/metrics_evolution/cnn_model_val_loss.csv', 'CNN Loss Evolution', 'Loss')

    # plot accuracy evolution of CNN model
    plot_train_test_metric('output/metrics_evolution/cnn_model_acc.csv',
                           'output/metrics_evolution/cnn_model_val_acc.csv', 'CNN Accuracy Evolution', 'Accuracy')
    """
    plot_sample(model='cnn', phase='ES')
