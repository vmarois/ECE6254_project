# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 03/04/2018
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, UpSampling2D, ZeroPadding2D, Lambda
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import load_model
from keras import backend as K

import os
import numpy as np
from data import load_data, load_data_seg

#   PARAMETERS  #
num_classes = 4  # 4 target features to output
epochs = 400  # number of training epochs (on full dataset)
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
    print('\Model Summary:\n')
    print(model.summary())

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
    print('\Model Summary:\n')
    print(model.summary())

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


def train_dnn_model_kfold():
    """
    Train the DNN model doing 5-fold cross-validation. Don't save the model to file.
    """

    print('#' * 30)
    print('DNN Model Kfold training routine: ')
    # get training data
    X_train, y_train = load_data(model='dnn', set='train', img_rows=img_rows, img_cols=img_cols)

    # load test data
    X_test, y_test = load_data(model='dnn', set='test', img_rows=img_rows, img_cols=img_cols)

    # create masks for k fold cross validation
    mask1 = np.zeros((len(X_train)), dtype=bool)
    mask1[:500] = True
    mask2 = np.zeros((len(X_train)), dtype=bool)
    mask2[500:1000] = True
    mask3 = np.zeros((len(X_train)), dtype=bool)
    mask3[1000:1500] = True
    mask4 = np.zeros((len(X_train)), dtype=bool)
    mask4[1500:2000] = True
    mask5 = np.zeros((len(X_train)), dtype=bool)
    mask5[2000:] = True

    splits = [(True^mask1, mask1), (True^mask2, mask2), (True^mask3, mask3), (True^mask4, mask4), (True^mask5, mask5)]

    accs = []
    losses = []
    val_accs = []
    val_losses = []
    test_scores = []

    for idx, (train, test) in enumerate(splits):
        print('\nFold n° ', idx+1)

        # create model
        model = dnn_model()

        # fit model on training data: train on 4 folds, test on the last one
        print('Fitting model on training data:')
        history = model.fit(X_train[train], y_train[train], batch_size=64, epochs=epochs, shuffle=True, verbose=1,
                            validation_data=(X_train[test], y_train[test]))

        # evaluate the model on the (true) test data
        score = model.evaluate(X_test, y_test, verbose=1)
        print('Test mean squared error:', score[0])
        print('Test accuracy:', score[1])
        test_scores.append(score[1])

        # save metrics evolution to file
        accs.append(history.history['acc'])
        losses.append(history.history['loss'])
        val_accs.append(history.history['val_acc'])
        val_losses.append(history.history['val_loss'])

    print('\n5-fold cross validation done.')
    # compute average & std on the 5 folds
    mean_acc = np.mean(np.array(accs), axis=0)
    std_acc = np.std(np.array(accs), axis=0)

    mean_loss = np.mean(np.array(losses), axis=0)
    std_loss = np.std(np.array(losses), axis=0)

    mean_val_acc= np.mean(np.array(val_accs), axis=0)
    std_val_acc = np.std(np.array(val_accs), axis=0)

    mean_val_losses = np.mean(np.array(val_losses), axis=0)
    std_val_losses = np.std(np.array(val_losses), axis=0)

    mean_test_acc = np.mean(test_scores)*100
    std_test_acc = np.std(test_scores)*100
    print('\nAverage test accuracy: %.2f%% (+/- %.2f%%)' % (mean_test_acc, std_test_acc))

    # Create directory to store metrics evolution to file.
    directory = os.path.join(os.getcwd(), 'output/metrics_evolution/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save mean metrics evolution to file
    np.savetxt('output/metrics_evolution/dnn_mean_loss.csv', mean_loss)
    np.savetxt('output/metrics_evolution/dnn_mean_acc.csv', mean_acc)
    np.savetxt('output/metrics_evolution/dnn_mean_val_acc.csv', mean_val_acc)
    np.savetxt('output/metrics_evolution/dnn_mean_val_loss.csv', mean_val_losses)

    # save std metrics evolution
    np.savetxt('output/metrics_evolution/dnn_std_loss.csv', std_loss)
    np.savetxt('output/metrics_evolution/dnn_std_acc.csv', std_acc)
    np.savetxt('output/metrics_evolution/dnn_std_val_acc.csv', std_val_acc)
    np.savetxt('output/metrics_evolution/dnn_std_val_loss.csv', std_val_losses)

    print('\nSaved metrics evolution during training to file.')


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


def train_cnn_model_kfold():
    """
    Train the DNN model doing 5-fold cross-validation. Don't save the model to file.
    """

    print('#' * 30)
    print('CNN Model 5fold training routine: ')
    # get training data
    X_train, y_train = load_data(model='cnn', set='train', img_rows=img_rows, img_cols=img_cols)

    # load test data
    X_test, y_test = load_data(model='cnn', set='test', img_rows=img_rows, img_cols=img_cols)

    # create masks for k fold cross validation
    mask1 = np.zeros((len(X_train)), dtype=bool)
    mask1[:500] = True
    mask2 = np.zeros((len(X_train)), dtype=bool)
    mask2[500:1000] = True
    mask3 = np.zeros((len(X_train)), dtype=bool)
    mask3[1000:1500] = True
    mask4 = np.zeros((len(X_train)), dtype=bool)
    mask4[1500:2000] = True
    mask5 = np.zeros((len(X_train)), dtype=bool)
    mask5[2000:] = True

    splits = [(True ^ mask1, mask1), (True ^ mask2, mask2), (True ^ mask3, mask3), (True ^ mask4, mask4),
              (True ^ mask5, mask5)]

    accs = []
    losses = []
    val_accs = []
    val_losses = []
    test_scores = []

    for idx, (train, test) in enumerate(splits):
        print('\nFold n° ', idx + 1)

        # initialize dynamic change of learning rate : We start at the value of 'lr_start_cnn' and decrease it every
        # epoch to get to the final value of 'lr_stop_cnn'
        # initialize early stop : stop training if the monitored metric does not change for 'patience' epochs
        learning_rate = np.linspace(lr_start_cnn, lr_stop_cnn, epochs)
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(monitor='loss', patience=10)

        # create model
        model = cnn_model()

        # fit model on training data
        print('Fitting model on training data:')
        history = model.fit(X_train[train], y_train[train], batch_size=64, epochs=epochs,
                            callbacks=[change_lr, early_stop], validation_data=(X_train[test], y_train[test]),
                            shuffle=True, verbose=1)

        # evaluate the model on the (true) test data
        score = model.evaluate(X_test, y_test, verbose=1)
        print('Test mean squared error:', score[0])
        print('Test accuracy:', score[1])
        test_scores.append(score[1])

        # save metrics evolution to file
        accs.append(history.history['acc'])
        losses.append(history.history['loss'])
        val_accs.append(history.history['val_acc'])
        val_losses.append(history.history['val_loss'])

    print('\n5-fold cross validation done.')
    # compute average & std on the 5 folds
    mean_acc = np.mean(np.array(accs), axis=0)
    std_acc = np.std(np.array(accs), axis=0)

    mean_loss = np.mean(np.array(losses), axis=0)
    std_loss = np.std(np.array(losses), axis=0)

    mean_val_acc = np.mean(np.array(val_accs), axis=0)
    std_val_acc = np.std(np.array(val_accs), axis=0)

    mean_val_losses = np.mean(np.array(val_losses), axis=0)
    std_val_losses = np.std(np.array(val_losses), axis=0)

    mean_test_acc = np.mean(test_scores) * 100
    std_test_acc = np.std(test_scores) * 100
    print('\nAverage test accuracy: %.2f%% (+/- %.2f%%)' % (mean_test_acc, std_test_acc))

    # Create directory to store metrics evolution to file.
    directory = os.path.join(os.getcwd(), 'output/metrics_evolution/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save mean metrics evolution to file
    np.savetxt('output/metrics_evolution/cnn_mean_loss.csv', mean_loss)
    np.savetxt('output/metrics_evolution/cnn_mean_acc.csv', mean_acc)
    np.savetxt('output/metrics_evolution/cnn_mean_val_acc.csv', mean_val_acc)
    np.savetxt('output/metrics_evolution/cnn_mean_val_loss.csv', mean_val_losses)

    # save std metrics evolution
    np.savetxt('output/metrics_evolution/cnn_std_loss.csv', std_loss)
    np.savetxt('output/metrics_evolution/cnn_std_acc.csv', std_acc)
    np.savetxt('output/metrics_evolution/cnn_std_val_acc.csv', std_val_acc)
    np.savetxt('output/metrics_evolution/cnn_std_val_loss.csv', std_val_losses)

    print('\nSaved metrics evolution during training to file.')


if __name__ == '__main__':
    train_cnn_model(weights=False)
    train_dnn_model(weights=False)

    # use 5-fold cross validation (for learning curves plots)
    train_dnn_model_kfold()
    train_cnn_model_kfold()
