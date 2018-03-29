# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 27/03/2018
"""
import os
import scipy.ndimage.interpolation as interpol
from skimage import exposure
from skimage.transform import resize
from .acquisition import *


def data_augmentation_pipeline(img_rows=128, img_cols=128, rotation=True, shift=True, flip=True, contrast=True):
    """
    Loop over the data/ directory to retrieve each image & its associated ground truth segmentation mask.
    Apply the data augmentation techniques if the corresponding flag is true:
    :param rotation: flag to apply or not rotation on the dataset. If yes:
        - Use the same rotation angle for images of a same patient. Randomly select a rotation angle in [-10°, +10°]
    :param shift: flag to apply or not shifting on the dataset. If yes:
        - Use the same shift offset for images of a same patient. Randomly select an offset in [-10, +10]
    :param flip: flag to apply or not flipping (vertically) on the dataset.
    :param contrast: flag to apply or not contrast stretching on the dataset. If yes:
        - Increase contrast of the images using the contrast stretching method with the 4th & 96th percentiles as cutoff
    points. See http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm for more detail. No need to apply this modification
    on the segmentation masks.

    Resizes the images & masks to (img_rows, img_cols).
    The images & their variants are stored into different .npy files.
    """
    # get the patients ids located in the data/ directory
    patients = [name for name in os.listdir(os.path.join(os.curdir, 'data/')) if not name.startswith('.')]
    # sort this list to get the patients id in increasing order
    patients.sort(key=lambda s: s[-3:])  # sort according to last 3 characters
    images = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)  # 2 images per patient
    masks = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)  # 2 masks per patient

    print('Data Augmentation Pipeline - Considering the following transformations:')

    if rotation:
        print(' Rotation by a random angle within [-10°, +10°]')
        # range of rotation angles
        rot_angles = np.arange(-10, 11, 1)
        rotated_images = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)
        rotated_masks = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)

    if shift:
        print(' Shifting by a random offset within [-10, +10]')
        # range of offsets
        offsets = np.arange(-30, 31, 1)
        shifted_images = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)
        shifted_masks = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)

    if flip:
        print(' Flipping vertically')
        flipped_images = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)
        flipped_masks = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)

    if contrast:
        print(' Contrast stretching with the 4th & 96th percentiles as cutoff points')
        contrast_images = np.ndarray((2 * len(patients), img_rows, img_cols), dtype=np.uint8)

    # we now go through each patient's directory :
    idx = 0
    for patient in patients:
        print(' patient {}'.format(idx//2))

        if rotation:
            # randomly select a rotation angle from the specified range for each patient
            rot_angle = np.random.choice(rot_angles)
        if shift:
            # randomly select a offset from the specified range for each patient
            shift_offset = np.random.choice(offsets)

        for phase in ['ED', 'ES']:

            # read image & mask
            img, _, _, _ = load_mhd_data('data/{pa}/{pa}_4CH_{ph}.mhd'.format(pa=patient, ph=phase))
            mask, _, _, _ = load_mhd_data('data/{pa}/{pa}_4CH_{ph}_gt.mhd'.format(pa=patient, ph=phase))

            if rotation:
                # resize first, then rotate image & mask by same angle

                rotated_img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
                rotated_mask = resize(mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

                rotated_img = interpol.rotate(rotated_img, rot_angle, axes=(1, 0), reshape=False, output=None, order=1, mode='constant',
                                      cval=0.0, prefilter=True)  # uses simple interpolation (deg 1)
                rotated_mask = interpol.rotate(rotated_mask, rot_angle, axes=(1, 0), reshape=False, output=None, order=1, mode='constant',
                                       cval=0.0, prefilter=True)

                rotated_images[idx] = rotated_img
                rotated_masks[idx] = rotated_mask

            if shift:

                shifted_img = interpol.shift(img, shift_offset, output=None, order=3, mode='constant', cval=0.0,
                                             prefilter=True)
                shifted_mask = interpol.shift(mask, shift_offset, output=None, order=3, mode='constant', cval=0.0,
                                              prefilter=True)

                shifted_img = resize(shifted_img, (img_cols, img_rows), mode='reflect', preserve_range=True)
                shifted_mask = resize(shifted_mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

                shifted_images[idx] = shifted_img
                shifted_masks[idx] = shifted_mask

            if flip:
                flipped_img = np.flip(img, axis=1)
                flipped_mask = np.flip(mask, axis=1)

                flipped_img = resize(flipped_img, (img_cols, img_rows), mode='reflect', preserve_range=True)
                flipped_mask = resize(flipped_mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

                flipped_images[idx] = flipped_img
                flipped_masks[idx] = flipped_mask

            if contrast:
                # Contrast stretching
                p4, p96 = np.percentile(img, (4, 96))
                img_rescale = exposure.rescale_intensity(img, in_range=(p4, p96))

                img_rescale = resize(img_rescale, (img_cols, img_rows), mode='reflect', preserve_range=True)

                contrast_images[idx] = img_rescale

            # resize the img & the mask to (img_rows, img_cols) to keep the network input manageable
            img = resize(img, (img_cols, img_rows), mode='reflect', preserve_range=True)
            mask = resize(mask, (img_cols, img_rows), mode='reflect', preserve_range=True)

            images[idx] = img
            masks[idx] = mask

            idx += 1

    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/augmented_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save all ndarrays to a .npy files (for faster loading later)
    np.save('output/augmented_data/images.npy', images)
    np.save('output/augmented_data/masks.npy', masks)

    if rotation:
        np.save('output/augmented_data/rotated_images.npy', rotated_images)
        np.save('output/augmented_data/rotated_masks.npy', rotated_masks)

    if shift:
        np.save('output/augmented_data/shifted_images.npy', shifted_images)
        np.save('output/augmented_data/shifted_masks.npy', shifted_masks)

    if flip:
        np.save('output/augmented_data/flipped_images.npy', flipped_images)
        np.save('output/augmented_data/flipped_masks.npy', flipped_masks)

    if contrast:
        np.save('output/augmented_data/contrast_images.npy', contrast_images)

    print('Data augmentation by rotation done.')


def rotate_dataset(images, masks):
    """
    Apply a rotation to the dataset (images & associated segmentation masks).
    Use the same rotation angle for images of a same patient
    Randomly select an angle in [-10°, +10°]
    :param images: .npy file containing the (resized) images
    :param masks: .npy file containing the (resized) masks
    :return: rotated_img, rotated_masks saved to file
    """
    orig_images = np.load(images)
    orig_masks = np.load(masks)

    rot_angle = np.arange(-10, 11, 1)  # range of rotation angle

    rotated_images = np.ndarray(orig_images.shape, dtype=np.uint8)
    rotated_masks = np.ndarray(orig_masks.shape, dtype=np.uint8)

    for idx in range(0, orig_images.shape[0], 2):

        # randomly select a rotation angle from the specified range
        angle = np.random.choice(rot_angle)

        rotated_images[idx] = interpol.rotate(orig_images[idx], angle, axes=(1, 0), reshape=False, output=None, order=3, mode='constant',
                                      cval=0.0, prefilter=True)  # uses spline interpolation (deg 3)

        rotated_images[idx+1] = interpol.rotate(orig_images[idx+1], angle, axes=(1, 0), reshape=False, output=None, order=3,
                                              mode='constant', cval=0.0, prefilter=True)

        rotated_masks[idx] = interpol.rotate(orig_masks[idx], angle, axes=(1, 0), reshape=False, output=None, order=3, mode='constant',
                                      cval=0.0, prefilter=True)

        rotated_masks[idx+1] = interpol.rotate(orig_masks[idx+1], angle, axes=(1, 0), reshape=False, output=None, order=3,
                                             mode='constant',
                                             cval=0.0, prefilter=True)

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
    :return: shifted_img, shifted_masks saved to file
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


def contrast_stretching_dataset(images, low_p, high_p):
    """
    Increase contrast of the images using the contrast stretching method with the 4th & 96th percentiles as cutoff
    points. See http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm for more detail.
    No need to apply this modification on the segmentation masks.
    :param images: .npy file containing the (resized) images
    :param low_p: lower percentile
    :param high_p: higher percentile
    :return: contrast_images saved to file.
    """
    orig_images = np.load(images)

    contrast_images = np.ndarray(orig_images.shape, dtype=np.uint8)

    for idx, img in enumerate(orig_images):
        # Contrast stretching
        lp, up = np.percentile(img, (low_p, high_p))
        img_rescale = exposure.rescale_intensity(img, in_range=(lp, up))
        contrast_images[idx] = img_rescale

    # Create directory to store files.
    directory = os.path.join(os.getcwd(), 'output/augmented_data/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save ndarray to a .npy file (for faster loading later)
    np.save('output/augmented_data/contrast_images.npy', contrast_images)
    print('Data augmentation by contrast stretching done.')