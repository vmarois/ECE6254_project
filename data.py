# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 27/03/2018
"""
import os
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from helpers import *


def create_dataset(img_rows=128, img_cols=128):
    """
    Loop over the data/ directory to retrieve each image & its associated ground truth segmentation mask.
    Resizes the images & masks to (img_rows, img_cols) & stores them into 2 np.ndarrays.
    Also writes these 2 np.ndarrays to .npy files for faster loading when reusing them.
    :return: images np.ndarrays, masks np.ndarray
    """
    print('Creating original dataset from the raw data')
    # first, get the patients directory names located in the data/ directory. These names (e.g. 'patient0001') will
    # be used for indexing (also avoid hidden files & folders)
    patients = [name for name in os.listdir(os.path.join(os.curdir, 'data/')) if not name.startswith('.')]

    # We sort this list to get the patients id in increasing order
    patients.sort(key=lambda s: s[-3:])  # sort according to last 3 characters

    # create an empty numpy.ndarray which will contain the images (resized to (img_rows, img_cols))
    images = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)  # 2 images per patient
    masks = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)  # 2 masks per patient

    # we now go through each patient's directory :
    idx = 0
    for patient in patients:

        for phase in ['ED', 'ES']:

            # read image & mask
            img, _, _, _ = load_mhd_data('data/{pa}/{pa}_4CH_{ph}.mhd'.format(pa=patient, ph=phase))
            mask, _, _, _ = load_mhd_data('data/{pa}/{pa}_4CH_{ph}_gt.mhd'.format(pa=patient, ph=phase))

            # resize the img & the mask to (img_rows, img_cols) to keep the network input manageable
            img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
            mask = resize(mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

            # now, save the resized image to the images np.ndarray
            images[idx] = img

            # save the corresponding mask to masks np.ndarray (at the same index)
            masks[idx] = mask

            idx += 1

    print('Created 2 np.ndarrays containing images & masks.')

    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/processed_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save all ndarrays to a .npy files (for faster loading later)
    np.save('output/processed_data/images.npy', images)
    np.save('output/processed_data/masks.npy', masks)
    print('Saving to .npy files done: see files\noutput/processed_data/images.npy & \noutput/processed_data/masks.npy.')


def concatenate_datasets(filenames_list, img_rows=128, img_cols=128):
    """
    Concatenate the datasets (.npy files) contained in output/augmented_data into 1.
    We also use the segmentation masks to compute the center & main orientation of the left ventricle

    :param filenames_list: list of tuples specifying the pairs of images & masks:
        [(images, masks), (rotated_images, rotated_masks)..]
    :param img_rows, img_cols: images dimensions
    :return: whole set of images + ground truth values for center, orientation saved to .npy files
    """
    print('Concatenating the datasets created by data augmentation into a single one')
    print('Using the following pairs of images / masks datasets: ')
    print(filenames_list)
    print('\n')

    # total number of images
    n_samples = 600 * len(filenames_list)

    # create np.ndarrays for the images and the targets: xCenter, yCenter, xOrientation, yOrientation
    images_dataset = np.ndarray((n_samples, 128, 128), dtype=np.uint8)
    targets_dataset = np.ndarray((n_samples, 4), dtype=np.float32)

    for ds, (img, mask) in enumerate(filenames_list):
        print(" Processing {}".format(img))
        images = np.load("output/augmented_data/{}.npy".format(img))
        masks = np.load("output/augmented_data/{}.npy".format(mask))

        for idx, mat in enumerate(masks):

            # get the center coordinates of the left ventricle (on the resized image)
            row, col = findCenter(img=mat, pixelvalue=1)

            # get the orientation of the left ventricle (on the resized image)
            x_v1, y_v1 = findMainOrientation(img=mat, pixelvalue=1)

            # save the center coordinates & orientation to the y dataframe (which will be the output of the network)
            targets_dataset[ds*600 + idx] = np.array([row, col, x_v1, y_v1])

            # save image in main dataset file
            images_dataset[ds*600 + idx] = images[idx]

    print('Concatenated all datasets into one & created target values for (center, orientation)')

    print('Splitting the dataset into 70% training & 30% testing')
    images_train, images_test, targets_train, targets_test = train_test_split(images_dataset, targets_dataset,
                                                                              test_size=0.3,
                                                                              random_state=42,
                                                                              shuffle=True)

    # save all ndarrays to a .npy files (for faster loading later)
    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/processed_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save training set to file
    np.save('output/processed_data/images_train.npy', images_train)
    np.save('output/processed_data/targets_train.npy', targets_train)

    # save testing set to file
    np.save('output/processed_data/images_test.npy', images_test)
    np.save('output/processed_data/targets_test.npy', targets_test)
    print('Saving to .npy files done. See files: ')
    print('output/processed_data/images_train.npy')
    print('output/processed_data/targets_train.npy')
    print('output/processed_data/images_test.npy')
    print('output/processed_data/targets_test.npy')


def concatenate_datasets_seg(filenames_list, img_rows=128, img_cols=128):
    """
    Concatenate the datasets (.npy files) contained in output/augmented_data into 1 for the segmentation net.

    :param filenames_list: list of tuples specifying the pairs of images & masks:
        [(images, masks), (rotated_images, rotated_masks)..]
    :param img_rows, img_cols: images dimensions
    :return: whole set of images + ground truth values for center, orientation saved to .npy files
    """

    print('Concatenating the datasets created by data augmentation into a single one')
    print('Using the following pairs of images / masks datasets: ')
    print(filenames_list)
    print('\n')

    # total number of images
    n_samples = 600 * len(filenames_list)

    # create np.ndarrays for the images and the targets: xCenter, yCenter, xOrientation, yOrientation
    images_dataset = np.ndarray((n_samples, 128, 128), dtype=np.uint8)
    targets_dataset = np.ndarray((n_samples, 128, 128), dtype=np.uint8)

    for ds, (img, mask) in enumerate(filenames_list):
        print(" Processing {}".format(img))
        images = np.load("output/augmented_data/{}.npy".format(img))
        masks = np.load("output/augmented_data/{}.npy".format(mask))

        for idx, mat in enumerate(masks):

            # save mask in main dataset file
            targets_dataset[ds*600 + idx] = mat

            # save image in main dataset file
            images_dataset[ds*600 + idx] = images[idx]

    print('Concatenated all datasets into one & created target values for (center, orientation)')

    print('Splitting the dataset into 70% training & 30% testing')
    images_train, images_test, targets_train, targets_test = train_test_split(images_dataset, targets_dataset,
                                                                              test_size=0.3,
                                                                              random_state=42,
                                                                              shuffle=True)

    # save all ndarrays to a .npy files (for faster loading later)
    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/processed_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save training set to file
    np.save('output/processed_data/seg_images_train.npy', images_train)
    np.save('output/processed_data/seg_targets_train.npy', targets_train)

    # save testing set to file
    np.save('output/processed_data/seg_images_test.npy', images_test)
    np.save('output/processed_data/seg_targets_test.npy', targets_test)
    print('Saving to .npy files done. See files: ')
    print('output/processed_data/seg_images_train.npy')
    print('output/processed_data/seg_targets_train.npy')
    print('output/processed_data/seg_images_test.npy')
    print('output/processed_data/seg_targets_test.npy')


def load_data(model, set='train', img_rows=128, img_cols=128):
    """
    Loading training data & doing some additional preprocessing on it. If the indicated model is a dnn, we flatten out
    the input images. If the indicated model is a cnn, we put the channels first.
    :param model: string to indicate the type of model to prepare the data for. Either 'dnn' or 'cnn'
    :param set: string to specify whether we load the training or testing data
    :param img_rows: the new x-axis dimension used to resize the images
    :param img_cols: the new y-axis dimension used to resize the images
    :return: images & target features as numpy arrays.
    """
    print('#' * 30)
    print('Loading {} data from file.'.format(set))

    # read in the .npy file containing the images
    images_train = np.load('output/processed_data/images_{}.npy'.format(set))

    # read in the .npy file containing the target features
    targets_train = np.load('output/processed_data/targets_{}.npy'.format(set))

    # scale image pixel values to [0, 1]
    images_train = images_train.astype(np.float32)
    images_train /= 255.

    # scale target center coordinates to [-1, 1] (from 0 to 95 initially)
    targets_train = targets_train.astype(np.float32)
    targets_train[:, 0] = (targets_train[:, 0] - (img_rows / 2)) / (img_rows / 2)
    targets_train[:, 1] = (targets_train[:, 1] - (img_rows / 2)) / (img_cols / 2)

    # reshape images according to the neural network model intended to be used
    if model == 'cnn':
        print('Indicated model is a CNN, reshaping images with channels first.')
        images_train = images_train.reshape(-1, 1, img_rows, img_cols)
    elif model == 'dnn':
        print('Indicated model is a DNN, flattening out images.')
        images_train = images_train.reshape(images_train.shape[0], img_rows * img_rows)

    print('Loading done. Pixel values have been scaled to [0, 1] and target center coordinates to [-1, 1].')
    print('#' * 30)

    return images_train, targets_train


def load_data_seg(set='train', img_rows=128, img_cols=128):
    """
    Loading training data for the Segmentation CNN & doing some additional preprocessing on it. Putting the channels first.
    :param set: string to specify whether we load the training or testing data
    :param img_rows: the new x-axis dimension used to resize the images
    :param img_cols: the new y-axis dimension used to resize the images
    :return: images & target features as numpy arrays.
    """
    print('#' * 30)
    print('Loading {} data from file.'.format(set))

    # read in the .npy file containing the images
    images_train = np.load('output/processed_data/seg_images_{}.npy'.format(set))

    # read in the .npy file containing the target features
    targets_train = np.load('output/processed_data/seg_targets_{}.npy'.format(set))

    # scale image pixel values to [0, 1]
    images_train = images_train.astype(np.float32)
    images_train /= 255.

    # reshape images according to the neural network model intended to be used
    print('Reshaping images with channels first.')
    images_train = images_train.reshape(-1, 1, img_rows, img_cols)

    print('Loading done. Pixel values have been scaled to [0, 1] and target center coordinates to [-1, 1].')
    print('#' * 30)

    return images_train, targets_train


if __name__ == '__main__':
    #create_dataset(img_rows=128, img_cols=128)

    #data_augmentation_pipeline(img_rows=128, img_cols=128,rotation=True,shift=True,flip=True,contrast=True,blur=True)

    filenames_list = [('images', 'masks'),
                      ('rotated_images', 'rotated_masks'),
                      ('shifted_images', 'shifted_masks'),
                      ('flipped_images', 'flipped_masks'),
                      ('contrast_images', 'masks'),
                      ('blurred_images', 'masks')]

    #concatenate_datasets(filenames_list=filenames_list)

    concatenate_datasets_seg(filenames_list, img_rows=128, img_cols=128)
