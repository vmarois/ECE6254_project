# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 27/03/2018
"""
import os
import numpy as np
from skimage.transform import resize
from helpers import *


def create_dataset(img_rows=128, img_cols=128):
    """
    Loop over the data/ directory to retrieve each image & its associated ground truth segmentation mask.
    Resizes the images & masks to (img_rows, img_cols) & stores them into 2 np.ndarrays.
    Also writes these 2 np.ndarrays to .npy files for faster loading when reusing them.
    :return: images np.ndarrays, masks np.ndarray
    """
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
    print('Saving to .npy files done.')


if __name__ == '__main__':
    create_dataset(img_rows=128, img_cols=128)
