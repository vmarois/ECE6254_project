# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 27/03/2018
"""
import os
import numpy as np
import scipy.ndimage.interpolation as interpol


def rotate_img(img, angle):
    """
    Rotate the specified img by the specified angle aroung the img center
    :param img: input img or mask
    :param angle: rotation angle
    :return: rotated image
    """
    img_rot = interpol.rotate(img, angle, axes=(1, 0), reshape=True, output=None, order=3, mode='constant',
                              cval=0.0, prefilter=True)  # uses spline interpolation (deg 3)
    return img_rot


def shift_img(img, shift):
    """
    Shift the specified img by the specified amount
    :param img: input img or mask
    :param shift: The shift along the axes. If a float, shift is the same for each axis. If a sequence, shift should
    contain one value for each axis.
    :return: shifted image
    """
    shifted_img = interpol.shift(img, shift, output=None, order=3, mode='constant', cval=0.0, prefilter=True)


def rotate_dataset(images, masks):
    """
    Apply a rotation to the original dataset (images & associated segmentation masks).
    Use the same rotation angle for images of a same patient
    Randomly select a rotation angle in [-10°, +10°]
    :param images: .npy file containing the (resized) images
    :param masks: .npy file containing the (resized) masks
    :return: rotated_img, rotated_masks saved to file
    """
    orig_images = np.load(images)
    orig_masks = np.load(masks)

    rot_angles = np.arange(-10, 11, 1)  # range of rotation angle

    rotated_images = np.ndarray(orig_images.shape, dtype=np.uint8)
    rotated_masks = np.ndarray(orig_masks.shape, dtype=np.uint8)

    for idx in range(0, orig_images.shape[0], 2):

        # randomly select a rotation angle from the specified range
        rot_angle = np.random.choice(rot_angles)

        rotated_images[idx] = interpol.rotate(orig_images[idx], rot_angle, axes=(1, 0), reshape=False, output=None, order=3, mode='constant',
                              cval=0.0, prefilter=True)  # uses spline interpolation (deg 3)

        rotated_images[idx+1] = interpol.rotate(orig_images[idx+1], rot_angle, axes=(1, 0), reshape=False, output=None,
                                              order=3, mode='constant',
                                              cval=0.0, prefilter=True)  # uses spline interpolation (deg 3)

        rotated_masks[idx] = interpol.rotate(orig_masks[idx], rot_angle, axes=(1, 0), reshape=False, output=None,
                                              order=3, mode='constant',
                                              cval=0.0, prefilter=True)  # uses spline interpolation (deg 3)

        rotated_masks[idx + 1] = interpol.rotate(orig_masks[idx + 1], rot_angle, axes=(1, 0), reshape=False,
                                                  output=None, order=3, mode='constant', cval=0.0,
                                                 prefilter=True)  # uses spline interpolation (deg 3)

    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/augmented_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save all ndarrays to a .npy files (for faster loading later)
    np.save('output/augmented_data/rotated_images.npy', rotated_images)
    np.save('output/augmented_data/rotated_masks.npy', rotated_masks)
    print('Data augmentation by rotation done.')


def shift_dataset(images, masks):
    """
    Apply a shift to the original dataset (images & associated segmentation masks).
    Use the same shift offset for images of a same patient
    Randomly select an offset in [-10, +10]
    :param images: .npy file containing the (resized) images
    :param masks: .npy file containing the (resized) masks
    :return: rotated_img, rotated_masks saved to file
    """
    orig_images = np.load(images)
    orig_masks = np.load(masks)

    shift_offset = np.arange(-10, 11, 1)  # range of rotation angle

    shifted_images = np.ndarray(orig_images.shape, dtype=np.uint8)
    shifted_masks = np.ndarray(orig_masks.shape, dtype=np.uint8)

    for idx in range(0, orig_images.shape[0], 2):

        # randomly select a rotation angle from the specified range
        shift = np.random.choice(shift_offset)

        shifted_images[idx] = interpol.shift(orig_images[idx], shift, output=None, order=3, mode='constant', cval=0.0,
                                             prefilter=True)

        shifted_images[idx+1] = interpol.shift(orig_images[idx+1], shift, output=None, order=3, mode='constant',
                                               cval=0.0, prefilter=True)

        shifted_masks[idx] = interpol.shift(orig_masks[idx], shift, output=None, order=3, mode='constant', cval=0.0,
                                            prefilter=True)

        shifted_masks[idx+1] = interpol.shift(orig_masks[idx+1], shift, output=None, order=3, mode='constant', cval=0.0,
                                              prefilter=True)

    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/augmented_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save all ndarrays to a .npy files (for faster loading later)
    np.save('output/augmented_data/shifted_images.npy', shifted_images)
    np.save('output/augmented_data/shifted_masks.npy', shifted_masks)
    print('Data augmentation by shifting done.')
