# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 11/04/2018
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

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


def plot_train_test_metric_kfold(model, metric, title, metricname, nb_epochs):
    """
    Create a plot showing the evolution of the same metric evaluated during epochs on train set & holdout set for
    k-fold cross validation.

    :param model: the model to use, either 'cnn' or 'dnn'
    :param metric: which metric to plot: either 'acc' or 'loss'
    :param title: string for the plot title & output filename.
    :param metricname: string for ylabel
    :param nb_epochs: number of epochs the model has been trained on

    :return: None, plot saved to file.
    """
    # load specified metric from specified model from file
    mean_metric = np.loadtxt('output/metrics_evolution/{}_mean_{}.csv'.format(model, metric))
    std_metric = np.loadtxt('output/metrics_evolution/{}_std_{}.csv'.format(model, metric))
    # same for validation data
    mean_val_metric = np.loadtxt('output/metrics_evolution/{}_mean_val_{}.csv'.format(model, metric))
    std_val_metric = np.loadtxt('output/metrics_evolution/{}_std_val_{}.csv'.format(model, metric))

    plt.style.use('ggplot')
    plt.grid(b=True)

    plt.plot(mean_metric, color='crimson', label='train')
    plt.fill_between(range(1, nb_epochs+1), mean_metric - std_metric, mean_metric + std_metric, alpha=0.1, color="crimson")

    plt.plot(mean_val_metric, color='mediumseagreen', label='test')
    plt.fill_between(range(1, nb_epochs+1), mean_val_metric - std_val_metric, mean_val_metric + std_val_metric, alpha=0.1, color="mediumseagreen")
    plt.title(title)
    plt.ylabel(metricname)
    plt.xlabel('epoch')
    plt.legend(loc='best')

    # Create directory to store pdf files.
    directory = os.path.join(os.getcwd(), 'output/plots/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('output/plots/{}.png'.format(title), bbox_layout='tight', dpi=300)
    plt.clf()

    print('{} plot saved to file.'.format(title))


def plot_sample(model):
    """
    Plot the predicted center & main orientation on a sample image randomly drawn from the test set.
    Also plot ground truth center & main orientation for reference

    :param model: the model to use, either 'cnn' or 'dnn'

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

    # compute distance between predicted & true center
    distance = np.sqrt((pred_row-true_row) ** 2 + (pred_col-true_col) ** 2)

    # compute angle difference between predicted orientation
    # compute vector corresponding to predicted orientation
    predVector = np.array((x_v1,y_v1))
    predMagnitude = np.linalg.norm(x=predVector, ord=2)

    # compute vector corresponding to ground truth orientation
    trueVector = np.array((true_x_v1,true_y_v1))
    trueMagnitude = np.linalg.norm(x=trueVector, ord=2)

    angle = np.dot(trueVector, predVector) / (trueMagnitude * predMagnitude)
    angle = angle % 1
    angle = math.acos(angle)
    angle = math.degrees(angle) % 360

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

    plt.text(0, 0, "dist: %.3f\nangle: %.3f"%(distance, angle),
             ha="left", va="top",
             bbox=dict(boxstyle="square",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       )
             )

    plt.axis('equal')
    plt.legend()
    plt.title('Predicted center & orientation vs GT.\n Model = {}'.format(model.upper()))

    # Create directory to store pdf files.
    directory = os.path.join(os.getcwd(), 'output/plots/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig("output/plots/sample_image_{m}_{i}.png".format(m=model.upper(), i=sample), bbox_inches='tight', dpi=300)
    print('Sample image plot saved to file.')
    plt.show()
    plt.clf()


def plot_data_augmentation():
    """
    Plot a random original image and the 5 artificially created corresponding images side-by-side for visualization.

    :return: plot saved to file.
    """
    filenames_list = ['images', 'rotated_images', 'shifted_images', 'flipped_images', 'contrast_images', 'blurred_images']

    plt.figure(figsize=(8, 6))

    sample = np.random.randint(600)

    for idx, file in enumerate(filenames_list):

        # load file
        print('Loading', file)
        images = np.load('output/augmented_data/{}.npy'.format(file))

        # create plot
        plt.subplot(2, 3, idx + 1)
        plt.imshow(images[sample], cmap='Greys_r')
        plt.axis('off')
        plt.title(file)

    plt.suptitle('Data Augmentation Example')
    plt.savefig('output/plots/data_augmentation_vis_{}.png'.format(sample), bbox_layout='tight', dpi=300)
    plt.show()
    plt.clf()


def vis_filters(layer='conv2d_1', img_width=128, img_height=128):
    """
    Try to visualize the filters of the specified convolutional filter.

    :param layer: layer name, e.g 'conv2d_1'
    :param img_height, img_width: input image size

    :return:
    """

    # load model
    model = load_model('output/models/cnn_model.h5')

    # the name of the layer we want to visualize
    layer_name = layer

    # util function to convert a tensor into a valid image
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_dim_ordering() == 'th':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[layer_name].output

    # utility function to normalize a tensor by its L2 norm
    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    kept_filters = []
    for filter_index in range(0, 32):  # 32 filters in first conv layer
        print('\nProcessing filter %d' % filter_index)

        # build a loss function that maximizes the activation of the ith filter of the layer considered
        loss = K.mean(layer_output[:, filter_index, :, :])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # start from a gray image with some random noise
        input_img_data = np.random.random((1, 1, img_width, img_height))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)

            # some filters get stuck to 0, we do not take them into account
            if loss_value <= 0.:
                print('Negative loss value, skipping current filter.\n')
                break

        if loss_value > 0:
            # decode the resulting input image
            img = deprocess_image(input_img_data[0])
            kept_filters.append(img)
        print('Filter %d processed.\n' % filter_index)

    nb_filters = len(kept_filters)
    print('Number of filters:', nb_filters)

    # Create a plot of the kept filters
    plt.figure(figsize=(8,8))

    for idx, filter in enumerate(kept_filters):
        plt.subplot(np.ceil(np.sqrt(nb_filters)), np.ceil(np.sqrt(nb_filters)), idx+1)
        plt.imshow(filter[0])
        plt.axis('off')
    plt.suptitle('Filters with loss > 0 for layer {}'.format(layer))
    plt.savefig('output/plots/filters_{}_layer.png'.format(layer), bbox_layout='tight', dpi=300)
    plt.show()
    plt.clf()
