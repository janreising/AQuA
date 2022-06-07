import dask.array as da
from dask.diagnostics import ProgressBar
from dask import compute

import tiledb

import h5py as h5
import numpy as np
import time
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

from skimage import measure, morphology  # , segmentation


def estimate_background(data: np.array, mask_xy: np.array = None) -> float:
    """ estimates overall noise level

    :param data: numpy.array
    :param mask_xy: binary 2D mask to ignore certain pixel
    :return: estimated standard error
    """

    xx = np.power(data[1:, :, :] - data[:-1, :, :], 2)  # dim: Z, X, Y
    stdMap = np.sqrt(np.median(xx, 0) / 0.9133)  # dim: X, Y

    if mask_xy is not None:
        stdMap[~mask_xy] = None

    stdEst = np.nanmedian(stdMap.flatten(), axis=0)  # dim: 1

    return stdEst


def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False):
    """Remove objects smaller than the specified size.

    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    in_place : bool, optional (default: False)
        If ``True``, remove the objects in the input array itself.
        Otherwise, make a copy.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, in_place=True)
    >>> d is a
    True

    """
    # Raising type error if not int or bool
    # _check_dtype_supported(ar)

    if in_place:
        out = ar
    else:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    # if len(component_sizes) == 2 and out.dtype != bool:
    #     warn("Only one label was provided to `remove_small_objects`. "
    #          "Did you mean to use a boolean array?")

    too_small = component_sizes < min_size
    print("too_small: ", too_small.shape)
    print("ccs: ", ccs.shape)
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def get_events(data: np.array, roi_threshold: float, var_estimate: float,
               min_roi_size: int = 10, mask_xy: np.array = None) -> (np.array, dict):
    """ identifies events in data based on threshold

    :param data: 3D array with dimensions Z, X, Y of dtype float
                expected to be photobleach corrected
    :param roi_threshold: minimum threshold to be considered an active pixel
    :param var_estimate: estimated variance of data
    :param min_roi_size: minimum size of active regions of interest
    :param mask_xy: (optional) 2D binary array masking pixels
    :return:
        event_map: 3D array in which pixels are labelled with event identifier
        event_properties: list of scipy.regionprops items
    """

    Z, X, Y = data.shape

    active_pixels = np.zeros(data.shape, dtype=np.bool8)

    # active_pixels = da.zeros(data.shape, dtype=bool)

    active_pixels[:] = data > roi_threshold * np.sqrt(var_estimate)
    print(active_pixels.dtype)
    morphology.remove_small_objects(active_pixels, min_size=min_roi_size, connectivity=4, in_place=True)

    # for z in range(Z):
    #
    #     img0 = img_thresh[z, :, :]
    #     # print(img0.dtype)
    #     img0 = morphology.remove_small_objects(img0, min_size=min_roi_size, connectivity=4)
    #
    #     if mask_xy is not None:
    #         img0 = np.multiply(img0, mask_xy)
    #
    #     active_pixels[z, :, :] = img0

    # event_map = measure.label(active_pixels)
    # event_properties = measure.regionprops(event_map, intensity_image=data)  # TODO split if necessary

    # return event_map, event_properties

    return active_pixels, None

if __name__ == "__main__":

    t0 = time.time()

    use_small = True
    use_dask = True

    # file path
    directory = "C:/Users/janrei/Desktop/"
    file = "22A5x4-1.zip.h5" if use_small else "22A5x4-2.zip.h5"
    loc = "/dff/neu" if use_small else "/dff/ast/"
    path = directory + file

    output = 'C:/Users/janrei/Desktop/22A5x4-2.zip.h5.tdb'

    # loading
    f = h5.File(path, "r")

    if use_dask:

        data = da.from_array(f[loc], chunks='auto')
        # data = tiledb.open(output)
        # data = da.from_array(data, chunks='auto')

        print(data)

        depth = {0: 5, 1: 5, 2: 5}

        active_pixels = data > 3 * np.sqrt(0.05)

        overlapping_grid = da.overlap.overlap(active_pixels, depth=depth, boundary='reflect')
        event_map = overlapping_grid.map_blocks(remove_small_objects, 10, 4, True)
        # event_map = event_map.map_blocks(measure.label)
        # event_map = event_map > 1
        event_map = da.overlap.trim_internal(event_map, depth)

        # event_map, event_properties = get_events(data, roi_threshold=3, var_estimate=0.05)

        # event_map, event_properties = compute(event_map, event_properties)

        # print("event_map.shape: ", event_map.shape)

    else:

        data = f[loc][:]
        # var_est = estimate_background(data)
        # print("Noise: ", var_est)
        event_map, event_properties = get_events(data, roi_threshold=3, var_estimate=0.05)
        # event_map = event_map > 1
        print("MAX: ", np.max(event_map))

    # print("Allclose: ", np.allclose(m1, m2))

    # x, y = 250, 250
    # # print("A: {}x{} {:.2f} {:.2f}".format(x, y, np.mean(m2[:, x, y]), np.mean(m1[:, x, y])))
    #
    # print("close? ", np.allclose(m1, m2))
    #
    # m3 = ndimage.uniform_filter1d(dat[:, x, y], size=25)
    #
    # fig, axx = plt.subplots(2, 2)
    # axx = axx.flatten()
    # axx[0].plot(dat[:, x, y])
    # axx[1].plot(m1[:, x, y], alpha=0.5, color="black", linestyle="-.")
    # axx[2].plot(m2[:, x, y], alpha=0.5, color="green", linestyle="--")
    # axx[3].plot(m3, alpha=0.3, color="red", linestyle="--")
    # plt.show()

    # plt.plot(m1[:, x, y], color="red")
    print("evt shape. ", event_map.shape)
    plt.imshow(event_map[0, :, :])
    plt.show()

    # clean up
    f.close()
    t1 = time.time()
    print("Runtime: {:.2f}".format(t1-t0))
