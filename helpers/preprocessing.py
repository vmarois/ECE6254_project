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
    This function returns the center coordinates of the different connected regions of an image.
    We consider only the pixels for which their value is equal to pixelvalue
    :param img: input image
    :param pixelvalue: int, specify which pixels to select.
    :return: ([x1, x2 ... xn], [y1, y2 ... yn]) where xi,yi are the coordinates of the ith region detected in the
    image (total of n regions). If only one region is detected, the 2 coordinates are returned as a tuple (x,y).
    """
    # use a boolean mask to select pixels of interest (intensity == pixelvalue)
    blobs = (img == pixelvalue)
    # label the n connected regions that satisfy this condition
    labels, nlabels = ndimage.label(blobs)
    # Find their unweighted centroids
    r, c = np.vstack(ndimage.center_of_mass(blobs, labels, np.arange(nlabels) + 1)).T  # returned as np.ndarray

    # round the values to int (since pixel coordinates)
    r = np.round(r).astype(int)
    c = np.round(c).astype(int)

    if nlabels == 1:  # if only 1 label, return a simple tuple
        return r[0], c[0]
    else:
        return r.tolist(), c.tolist()


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