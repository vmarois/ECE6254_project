# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 03/04/2018
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import backend as K

#   PARAMETERS  #
num_classes = 4  # 4 target features to output
epochs = 25  # number of training epochs (on full dataset)
lr_start_cnn = 0.01  # start value for decreasing learning rate (cnn model only)
lr_stop_cnn = 0.0005  # stop value for decreasing learning rate (cnn model only)
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

    return model


if __name__ == '__main__':
    cnn_model = cnn_model()