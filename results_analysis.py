# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 03/04/2018
"""
from helpers import *
from data import load_data

from keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os


def boxPlotDistance(img_rows=128, img_cols=128):
    """
    Create a seaborn boxplot comparing DNN & CNN models on the distribution of distance between the predicted center
    & the ground truth center.
    :return: None, create a seaborn boxplot.
    """
    distance = []
    label = []

    for net in ['dnn', 'cnn']:
        # load saved model
        model = load_model('output/models/{}_model.h5'.format(net))

        # load test data
        print('Loading {} test data'.format(net))
        images_test, targets_test = load_data(net, set='test')

        # get predictions
        net_pred = model.predict(images_test, verbose=1)

        # ground truth & predicted center coordinates are in [-1,1], scaling them back to [0, img_rows] to compute
        # the distance in pixels:
        for array in [net_pred, targets_test]:
            array[:, 0] = array[:, 0] * (img_rows / 2) + (img_rows / 2)
            array[:, 1] = array[:, 1] * (img_cols / 2) + (img_cols / 2)

        # compute distance between predicted center & true center and group result in a pandas dataframe
        net_distance = np.sqrt((targets_test[:, 0] - net_pred[:, 0]) ** 2 + (targets_test[:, 1] - net_pred[:, 1]) ** 2)
        print('{} average distance error (px): '.format(net.upper()), np.mean(net_distance))
        distance = np.concatenate((distance, net_distance))
        label += [net.upper()] * net_distance.shape[0]

    df = pd.DataFrame({'Distance (px)': distance, 'Model used': label})

    # generate seaborn boxplot
    plt.style.use('ggplot')
    plt.grid(b=True)
    sns.boxplot(x='Model used', y='Distance (px)', data=df, orient='v')
    plt.title('Distribution of predicted distance to true center.')

    # Create directory to store pdf files.
    directory = os.path.join(os.getcwd(), 'output/plots/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig("output/plots/distance_center_boxplot.png", bbox_inches='tight', dpi=300)
    plt.clf()


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
    #plot_sample(model='dnn', phase='ED')

    boxPlotDistance()
