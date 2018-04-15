# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 11/04/2018
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

img_rows, img_cols = 128, 128
scale = 35


def plot_train_test_metric(train, test, title, metricname):
    """
    Create a plot showing the evolution of the same metric evaluated during epochs on train set & holdout set.

    :param train: .csv filepath containing metric values evaluated on training data
    :param test: .csv filepath containing metric values evaluated on holdout data
    :param title: string for the plot title & output filename.
    :param metricname: plot ylabel.
    :return: None, plot saved to file.
    """
    metric = np.loadtxt(train)
    val_metric = np.loadtxt(test)

    plt.style.use('ggplot')
    plt.grid(b=True)
    plt.plot(metric)
    plt.plot(val_metric)
    plt.title(title)
    plt.ylabel(metricname)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')

    # Create directory to store pdf files.
    directory = os.path.join(os.getcwd(), 'output/plots/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('output/plots/{}.png'.format(title), bbox_layout='tight', dpi=300)
    plt.clf()

    print('{} plot saved to file.'.format(title))


def plot_sample(model, phase='ED'):
    """
    Plot the predicted center & main orientation on a sample image randomly drawn from the test set.
    Also plot ground truth center & main orientation for reference

    :param model: the model to use, either 'cnn' or 'dnn'
    :param phase: indicates which phase to select, either 'ED' or 'ES'

    :return: matplotlib.pyplot saved to file.
    """

    # load saved model
    saved_model = load_model('output/models/{}_model.h5'.format(model))

    # get random sample image from test set
    imgs = np.load('output/processed_data/images_test.npy')
    targets = np.load('output/processed_data/targets_test.npy')

    sample = np.random.randint(len(imgs))
    print('Sample', sample)
    img = imgs[sample]
    target = targets[sample]

    # get ground truth values
    true_row, true_col = target[0], target[1]
    true_x_v1, true_y_v1 = target[2], target[3]

    # scale image pixel values to [0, 1]
    img = img.astype(np.float32)
    img /= 255.

    # reshape input according to loaded model
    if model == 'dnn':
        inputimg = img.reshape(1, img_rows*img_cols)
    elif model == 'cnn':
        inputimg = img.reshape(-1, 1, img_rows, img_cols)

    # get prediction on input image
    prediction = saved_model.predict(inputimg, batch_size=1, verbose=1)

    # get target values (original scaling)
    pred_row = prediction[0, 0]*(img_rows/2) + (img_rows/2)
    pred_col = prediction[0, 1]*(img_cols/2) + (img_cols/2)
    x_v1 = prediction[0, 2]
    y_v1 = prediction[0, 3]

    # print some info
    print('True rowCenter, colCenter = ', true_row, true_col)
    print('Predicted rowCenter, colCenter = ', int(pred_row), int(pred_col))

    print('\nTrue xOrientation, yOrientation = ', true_x_v1, true_y_v1)
    print('Predicted xOrientation, yOrientation = ', x_v1, y_v1)

    # plot resized image
    plt.imshow(img, cmap='Greys_r')

    # plot predicted orientation line passing through predicted center
    plt.plot([pred_col - x_v1 * scale, pred_col + x_v1 * scale],
             [pred_row - y_v1 * scale, pred_row + y_v1 * scale],
             color='red', label='Prediction')

    # plot predicted orientation line passing through predicted center
    plt.plot([true_col - true_x_v1 * scale, true_col + true_x_v1 * scale],
             [true_row - true_y_v1 * scale, true_row + true_y_v1 * scale],
             color='black', label='Reference')

    fig = plt.gcf()
    ax = fig.gca()

    # plot predicted center
    pred_center = plt.Circle((pred_col, pred_row), 1, color='red', label='Prediction')
    ax.add_artist(pred_center)

    # plot true center
    true_center = plt.Circle((true_col, true_row), 1, color='black', label='Reference')
    ax.add_artist(true_center)

    plt.axis('equal')
    plt.legend()
    plt.title('Predicted center & orientation vs GT.\n Model = {} , Phase = {}'.format(model.upper(), phase))

    # Create directory to store pdf files.
    directory = os.path.join(os.getcwd(), 'output/plots/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig("output/plots/sample_image_{m}_{i}_{p}.png".format(m=model.upper(), i=sample, p=phase), bbox_inches='tight', dpi=300)
    print('Sample image plot saved to file.')
    plt.show()
    plt.clf()
