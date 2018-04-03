# This file is part of ECE6254_project
"""
@author: vmarois
@version: 1.0
@date: 02/04/2018
"""
import numpy as np
from scipy import ndimage


def getRoi(image, pixelvalue):
    """
    This function takes in an image (as a numpy 2D array), and return the regions composed of pixels with values equal
    to pixelvalue
    :param image: The image (numpy 2D array)
    :param pixelvalue: the pixel intensity value used to select the regions of interest
    :return: a numpy array of equal size to image, containing the specific region.
    """
    return ((image == pixelvalue)[:, :]).astype(int)


def findCenter(img, pixelvalue):
    """
    This function returns the center coordinates of the specified region of an image.
    We consider only the pixels for which their intensity is equal to pixelvalue
    :param img: input image
    :param pixelvalue: int, specify which pixels to select.
    :return: tuple (x,y).
    """
    # get coordinates of pixels of interest (intensity == pixelvalue)
    y, x = np.where(img == pixelvalue)

    # compute center coordinates as mean of the indices
    r = np.mean(y).astype(int)
    c = np.mean(x).astype(int)

    return r, c


def findMainOrientation(img, pixelvalue):
    """
    This function returns the main orientation of the region composed of pixels of the specified value.
    :param img: input image
    :param pixelvalue: the value used to filter the pixels
    :return: the x- & y-eigenvalues of the region as a tuple (essentially this is PCA, see
    https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/ for more info)
    """
    # get the indices of the pixels of value equal to pixelvalue
    y, x = np.where(img == pixelvalue)
    #  subtract mean from each dimension.
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    # covariance matrix and its eigenvectors and eigenvalues
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    # sort eigenvalues in decreasing order
    sort_indices = np.argsort(evals)[::-1]
    evec1, _ = evecs[:, sort_indices]

    # eigenvector with largest eigenvalue
    x_v1, y_v1 = evec1

    return x_v1, y_v1
