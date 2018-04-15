# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 03/04/2018
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, UpSampling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import load_model
from keras import backend as K

import os
import numpy as np
from data import load_data

#   PARAMETERS  #
num_classes = 4  # 4 target features to output
epochs = 50  # number of training epochs (on full dataset)
lr_start_cnn = 0.001  # start value for decreasing learning rate (cnn model only)
lr_stop_cnn = 0.0001  # stop value for decreasing learning rate (cnn model only)
lr_dnn = 0.0001

# input image dimensions
img_rows, img_cols = 128, 128
dnn_input_shape = img_rows * img_cols
cnn_input_shape = (1, img_rows, img_cols)
##################

K.set_image_data_format('channels_first')  # Sets the value of the data format convention.


def cnn_model():
    """
    Convolutional Neural Network Model.
    :return: compiled model.
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=cnn_input_shape))  # should output (32, 126, 126) as 128-3+1 = 126
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # # should output (32, 63, 63)

    model.add(Conv2D(64, (2, 2)))  # should output (64, 62, 62) as 63-2+1 = 62
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # should output (64, 31, 31)

    model.add(Conv2D(128, (2, 2)))  # should output (128, 30, 30) as 31-2+1 = 30
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # should output (128, 15, 15)

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))

    sgd = SGD(lr=lr_start_cnn, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd, metrics=['acc'])

    print('Created CNN model.')
    return model


def cnn_seg_model():
    """
    Convolutional Neural Network Model: same encoder as cnn_model(), but adds a decoder to predict segmentation mask.
    :return: compiled model.
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=cnn_input_shape))  # should output (32, 126, 126) as 128-3+1 = 126
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # # should output (32, 63, 63)

    model.add(Conv2D(64, (2, 2)))  # should output (64, 62, 62) as 63-2+1 = 62
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # should output (64, 31, 31)

    model.add(Conv2D(128, (2, 2)))  # should output (128, 30, 30) as 31-2+1 = 30
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # should output (128, 15, 15)

    # decoder
    model.add(UpSampling2D(size=(2, 2)))  # should output (128, 30, 30)
    model.add(ZeroPadding2D(padding=(1, 1)))   # should output (128, 32, 32)
    model.add(Conv2D(128, kernel_size=(2, 2)))  # should output (128, 31, 3) as 32-2+1 = 31

    model.add(UpSampling2D(size=(2, 2)))  # should output (128, 62, 62)
    model.add(ZeroPadding2D(padding=(1, 1)))  # should output (128, 64, 64)
    model.add(Conv2D(64, kernel_size=(2, 2)))  # should output (64, 63, 63) as 64-2+1 = 63

    model.add(UpSampling2D(size=(2, 2)))  # should output (64, 126, 126)
    model.add(ZeroPadding2D(padding=(2, 2)))  # should output (64, 128, 128)
    model.add(Conv2D(32, kernel_size=(2, 2)))  # should output (32, 127, 127)
    model.add(ZeroPadding2D(padding=(2, 2)))  # should output (32, 129, 129)
    model.add(Conv2D(4, kernel_size=(2, 2)))  # should output (4, 128, 128)

    sgd = SGD(lr=lr_start_cnn, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    print('Created CNN Segmentation model.')
    return model


def dnn_model():

    model = Sequential()

    model.add(Dense(100, input_dim=dnn_input_shape, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes))

    sgd = SGD(lr=lr_dnn, momentum=0.99, nesterov=True)

    model.compile(loss='mse', optimizer=sgd, metrics=['acc'])

    print('Created DNN model.')

    return model


def train_dnn_model(weights=True):
    """
    Train the DNN model.
    :param weights: If True, will load a saved model from file.
    :return: trained model saved to file.
    """

    print('#' * 30)
    print('DNN Model training routine: ')
    # get data
    X_train, y_train = load_data(model='dnn', set='train', img_rows=img_rows, img_cols=img_cols)

    # get model
    if weights:
        print(' Loading saved model from file.')
        model = load_model('output/models/dnn_model.h5')
    else:
        model = dnn_model()

    # fit model on training data
    print('Fitting model on training data:')
    hist = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_split=0.33, shuffle=True, verbose=1)

    # load test data
    X_test, y_test = load_data(model='dnn', set='test', img_rows=img_rows, img_cols=img_cols)

    # evaluate the model on the test data
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test mean squared error:', score[0])
    print('Test accuracy:', score[1])

    # Create directory to store metrics evolution to file.
    directory = os.path.join(os.getcwd(), 'output/metrics_evolution/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save metrics evolution
    np.savetxt('output/metrics_evolution/dnn_model_loss.csv', hist.history['loss'])
    np.savetxt('output/metrics_evolution/dnn_model_acc.csv', hist.history['acc'])
    np.savetxt('output/metrics_evolution/dnn_model_val_acc.csv', hist.history['val_acc'])
    np.savetxt('output/metrics_evolution/dnn_model_val_loss.csv', hist.history['val_loss'])
    print('Saved metrics evolution during training to file.')

    # Create directory to store model to file.
    directory = os.path.join(os.getcwd(), 'output/models/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    # save model

    model.save('output/models/dnn_model.h5')
    print('dnn model saved to .h5 file.')


def train_cnn_model(weights=True):
    """
    Train the CNN model.
    :param weights: If True, will load a saved model from file.
    :return: trained model saved to file.
    """

    print('#' * 30)
    print('CNN Model training routine: ')
    # load training data
    X_train, y_train = load_data(model='cnn', set='train', img_rows=img_rows, img_cols=img_cols)

    # get CNN model
    if weights:
        print(' Loading saved model from file.')
        model = load_model('output/models/cnn_model.h5')
    else:
        model = cnn_model()

    # initialize dynamic change of learning rate : We start at the value of 'lr_start_cnn' and decrease it every epoch
    # to get to the final value of 'lr_stop_cnn'
    # initialize early stop : stop training if the monitored metric does not change for 'patience' epochs
    learning_rate = np.linspace(lr_start_cnn, lr_stop_cnn, epochs)
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(monitor='loss', patience=10)

    # fit model on training data
    print('Fitting model on training data:')
    hist = model.fit(X_train, y_train, batch_size=64, epochs=epochs, callbacks=[change_lr, early_stop],
                     validation_split=0.33, shuffle=True, verbose=1)

    # load test data
    X_test, y_test = load_data(model='cnn', set='test', img_rows=img_rows, img_cols=img_cols)

    # evaluate the model on the test data
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test mean squared error:', score[0])
    print('Test accuracy:', score[1])

    # Create directory to store metrics evolution to file.
    directory = os.path.join(os.getcwd(), 'output/metrics_evolution/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save metrics evolution
    np.savetxt('output/metrics_evolution/cnn_model_loss.csv', hist.history['loss'])
    np.savetxt('output/metrics_evolution/cnn_model_acc.csv', hist.history['acc'])
    np.savetxt('output/metrics_evolution/cnn_model_val_acc.csv', hist.history['val_acc'])
    np.savetxt('output/metrics_evolution/cnn_model_val_loss.csv', hist.history['val_loss'])
    print('Saved metrics evolution during training to file.')

    # Create directory to store model to file.
    directory = os.path.join(os.getcwd(), 'output/models/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save model
    model.save('output/models/cnn_model.h5')
    print('CNN model saved to .h5 file.')


if __name__ == '__main__':

    train_cnn_model(weights=False)

    #train_dnn_model(weights=False)
