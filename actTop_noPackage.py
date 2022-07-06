import argparse
import os
import random

import h5py as h5
import tiledb
import numpy as np
import time
from itertools import product
import tempfile
from tqdm import tqdm
import gc

import scipy.ndimage as ndimage
from scipy import signal
from scipy.stats import zscore, norm
from scipy.optimize import curve_fit

from skimage import measure, morphology  # , segmentation

import dask.array as da
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler
from dask_image import ndmorph, ndfilters, ndmeasure

import matplotlib.pyplot as plt

# import multiprocessing as mp
# from multiprocessing import shared_memory

import multiprocess as mp
from multiprocess import shared_memory
from itertools import repeat

import sys
import traceback
import warnings

import dask

import json

def run(meta, file_path, dataset=None, threshold=3, min_size=20, moving_average=25, use_dask=False, adjust_for_noise=False, subset=None, output_folder=None):

    meta["subset"] = subset
    meta["threshold"] = threshold
    meta["min_size"] = min_size
    meta["adjust_for_noise"] = adjust_for_noise

    # profiling
    pbar = ProgressBar(minimum=5)
    pbar.register()

    # load data
    print("Loading data")
    data = _load(file_path, dataset_name=dataset, use_dask=use_dask, subset=subset)

    event_map_path = f"{output_folder}event_map/"
    if not os.path.isdir(event_map_path):
        print("Estimating noise")
        noise = estimate_background(data) if adjust_for_noise else 1

        print("Thresholding events")
        event_map = get_events(data, roi_threshold=threshold, var_estimate=noise, min_roi_size=min_size)

        del noise

        print("Saving event map to: ", event_map_path)
        if output_folder is not None:
            event_map.to_tiledb(event_map_path)

    else:
        print("Loading event map from: ", event_map_path)
        event_map = da.from_tiledb(event_map_path)

    # getting rid of extra variables
    del data
    gc.collect()

    print("Calculating time map")
    time_map_path = f"{output_folder}time_map.npy"
    if not os.path.isfile(time_map_path):
        time_map = get_time_map(event_map)

        print("Saving event map to: ", time_map_path)
        if output_folder is not None:
            np.save(time_map_path, time_map)
    else:
        print("Loading time map from: ", time_map_path)
        time_map = np.load(time_map_path)

    del event_map
    gc.collect()

    return None

    print("Calculating features")
    custom_slim_features(time_map, file_path, event_map_path)

    print("Saving")
    if output_folder is not None:
        with open(f"{output_folder}/meta.json", 'w') as outfile:
            json.dump(meta, outfile)

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


def _load(file_path, dataset_name: str = None, use_dask: bool = False, subset=None):

    """ loads data from file

    :param dataset_name: name of the dataset in stored in an hdf file
    :param in_memory: flag whether to load full dataset in memory or not
    :return:
    """

    if file_path.endswith(".h5"):

        assert dataset_name is not None, "'dataset_name' required if providing an hdf file"

        file = h5.File(file_path, "r")
        assert dataset_name in file, "dataset '{}' does not exist in file".format(dataset_name)

        data = da.from_array(file[dataset_name], chunks='auto') if use_dask else file[dataset_name]

    elif file_path.endswith(".tdb"):

        data = tiledb.open(file_path)

        if use_dask:
            data = da.from_array(data, chunks='auto')

    if subset is not None:
        assert len(subset) == 6, "please provide a subset for all dimensions"
        z0, z1, x0, x1, y0, y1 = subset

        z0 = z0 if z0 is not None else 0
        x0 = x0 if x0 is not None else 0
        y0 = y0 if y0 is not None else 0

        Z, X, Y = data.shape
        z1 = z1 if z1 is not None else Z
        x1 = x1 if x1 is not None else X
        y1 = y1 if y1 is not None else Y

        data = data[z0:z1, x0:x1, y0:y1]

    if not use_dask:
        data = data[:]

    return data

def get_events(data: np.array, roi_threshold: float, var_estimate: float,
               min_roi_size: int = 10, mask_xy: np.array = None, smoXY=1,
               remove_small_object_framewise=False) -> (np.array, dict):

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

    # threshold data by significance value
    active_pixels = da.from_array(np.zeros(data.shape, dtype=np.bool8))
    # self.vprint("Noise threshold: {:.2f}".format(roi_threshold * np.sqrt(var_estimate)), 4)
    absolute_threshold = roi_threshold * np.sqrt(var_estimate) if var_estimate is not None else roi_threshold
    active_pixels[:] = ndfilters.gaussian_filter(data, smoXY) > absolute_threshold

    # mask inactive pixels (accelerates subsequent computation)
    if mask_xy is not None:
        np.multiply(active_pixels, mask_xy, out=active_pixels)

    # subsequent analysis hard to parallelize; save in memory
    if remove_small_object_framewise:
        active_pixels = active_pixels.compute() if type(active_pixels) == da.core.Array else active_pixels

        # remove small objects
        for cz in range(active_pixels.shape[0]):
            active_pixels[cz, :, :] = morphology.remove_small_objects(active_pixels[cz, :, :],
                                                                      min_size=min_roi_size, connectivity=4)

        # label connected pixels
        # event_map, num_events = measure.label(active_pixels, return_num=True)
        event_map = np.zeros(data.shape, dtype=np.uint16)
        event_map[:], num_events = ndimage.label(active_pixels)

    else:

        # remove small objects
        struct = ndimage.generate_binary_structure(3, 4)

        active_pixels = ndmorph.binary_opening(active_pixels, structure=struct)
        active_pixels = ndmorph.binary_closing(active_pixels, structure=struct)

        # label connected pixels
        event_map = da.from_array(np.zeros(data.shape, dtype=np.uintc))
        event_map[:], num_events = ndimage.label(active_pixels)

    return event_map

def custom_slim_features(time_map_path, data_path, event_path):

        # print(event_map)
        # sh_em = shared_memory.SharedMemory(create=True, size=event_map.nbytes)
        # shn_em = np.ndarray(event_map.shape, dtype=event_map.dtype, buffer=sh_em.buf)
        # shn_em[:] = event_map
        #
        # num_events = np.max(shn_em)

        # create chunks

        time_map = np.load(time_map_path)

        print("\t collecting arguments")
        num_frames, num_rois = time_map.shape

        split_size = int(num_frames / (mp.cpu_count() * 10))
        splits = np.arange(0, num_frames, split_size)

        last_max_roi = 0
        c = []
        for split in splits:

            start, stop = split, split+split_size

            if stop >= num_frames:
                c.append([start, num_frames, last_max_roi, num_rois, len(splits)])
                continue

            last_frame = np.where(time_map[stop, :] == 1)[0]
            max_roi = np.max(last_frame)

            zs, _ = np.where(time_map[:, last_frame[1:]] == 1)
            max_z = np.max(zs)

            c.append([start, max_z, last_max_roi+1, max_roi, len(splits)])
            last_max_roi = max_roi

        random.shuffle(c)
        print("Num chunks: {} [{}]".format(len(c), len(c[0])))

        out_path = os.sep.join(event_path[:-1].split(os.sep)[:-1]) + "/events/"
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        arguments = zip(c, repeat(data_path), repeat(event_path), repeat(out_path))
        print("mp.cpu_count(): ", mp.cpu_count())

        with mp.Pool(mp.cpu_count()) as p:

            print("\t starting mapping")
            R = p.starmap(func_multi, arguments)

        # sh_em.close()
        # sh_em.unlink()

        return R

def func(event_id, shape, sh_event_name, file_path):

    from multiprocess import shared_memory
    import numpy as np
    import tiledb

    # load data
    em = shared_memory.SharedMemory(name=sh_event_name)
    em_np = np.ndarray(shape, dtype='uint16', buffer=em.buf)

    data = tiledb.open(file_path)

    res = {}
    res["label"] = event_id

    z, x, y = np.where(em_np == event_id)
    z0, z1 = np.min(z), np.max(z)
    x0, x1 = np.min(x), np.max(x)
    y0, y1 = np.min(y), np.max(y)

    res["area"] = len(z)
    res["bbox"] = ((z0, z1), (x0, x1), (y0, y1))

    dz, dx, dy = z1 - z0, x1 - x0, y1 - y0
    z, x, y = z - z0, x - x0, y - y0
    res["dim"] = (dz + 1, dx + 1, dy + 1)
    res["pix_num"] = int((dz+1)*(dx+1)*(dy+1))

    mask = np.ones((dz + 1, dx + 1, dy + 1), dtype=np.bool8)
    mask[(z, x, y)] = 0
    res["mask"] = mask.flatten()
    res["footprint"] = np.invert(np.min(mask, axis=0)).flatten()

    signal = data[z0:z1 + 1, x0:x1 + 1, y0:y1 + 1]  # TODO weird that i need +1 here
    msignal = np.ma.masked_array(signal, mask)
    res["trace"] = np.ma.filled(np.nanmean(msignal, axis=(1, 2)))

    # for mem in [em, d]:
    for mem in [em]:
        mem.close()
        mem.unlink()

    if event_id % 100 == 0:
        print("task {} finished!".format(event_id))

    return res

def func_multi(task, data_path, event_path, out_path):

    import numpy as np
    import tiledb

    t0, t1, r0, r1, task_num = task
    res_path = f"{out_path}events{r0}-{r1}.npy"
    print("Working on {}".format(res_path))

    if os.path.isfile(res_path):
        return 2

    # load data
    data = tiledb.open(data_path)[t0:t1]
    event_map = tiledb.open(event_path)[t0:t1]

    res = {}
    for event_id in range(r0, r1):

        res[event_id] = {}

        try:
            z, x, y = np.where(event_map == event_id)
            z0, z1 = np.min(z), np.max(z)
            x0, x1 = np.min(x), np.max(x)
            y0, y1 = np.min(y), np.max(y)

            res[event_id]["area"] = len(z)
            res[event_id]["bbox"] = ((z0, z1), (x0, x1), (y0, y1))

            dz, dx, dy = z1 - z0, x1 - x0, y1 - y0
            z, x, y = z - z0, x - x0, y - y0
            res[event_id]["dim"] = (dz + 1, dx + 1, dy + 1)
            res[event_id]["pix_num"] = int((dz+1)*(dx+1)*(dy+1))

            mask = np.ones((dz + 1, dx + 1, dy + 1), dtype=np.bool8)
            mask[(z, x, y)] = 0
            res[event_id]["mask"] = mask.flatten()
            res[event_id]["footprint"] = np.invert(np.min(mask, axis=0)).flatten()

            signal = data[z0:z1 + 1, x0:x1 + 1, y0:y1 + 1]  # TODO weird that i need +1 here
            msignal = np.ma.masked_array(signal, mask)
            res[event_id]["trace"] = np.ma.filled(np.nanmean(msignal, axis=(1, 2)))

            res[event_id]["error"] = 0

        except ValueError as err:
            print("\t Error in ", event_id)
            print("\t", err)
            res[event_id]["error"] = 1
            return -1

    np.save(res_path, res)

    num_files = len(os.listdir(out_path))
    print("Finished task {}-{} ({:.1f}%)".format(t0, t1, num_files/task_num*100))

    return 1

def get_time_map(event_map, chunk=200):

    time_map = np.zeros((event_map.shape[0], np.max(event_map) + 1), dtype=np.bool_)

    Z = event_map.shape[0]
    if type(event_map) == da.core.Array:

        print("Recognized dask array. Loading in chunks ...")
        for c in tqdm(range(0, Z, chunk)):

            cmax = min(Z, c+chunk)
            event_map_memory = event_map[c:cmax, :, :].compute()

            for z in range(c, cmax):
                time_map[z, np.unique(event_map_memory[z-c, :, :])] = 1

    else:

        print("Assuming event_map is in RAM ... ")
        for z in tqdm(range(Z)):
            time_map[z, np.unique(event_map[z, :, :])] = 1

    return time_map


if __name__ == "__main__":

    t0 = time.time()

    use_small = False
    use_dask = True
    use_subset = False

    subset = None if use_small or not use_subset else [0, 1000, None, None, None, None]
    print("subset: ", subset)

    # file path
    directory = "/home/janrei1@ad.cmm.se/Desktop/"
    file = "22A5x4-1.zip.h5" if use_small else "22A5x4-2-subtracted-mc.zip.h5.reconstructed_.h5.tdb"  # "22A5x4-2.zip.h5.tdb"
    loc = "/dff/neu" if use_small else "/dff/ast/"
    path = directory + file

    # run code
    meta = {}

    res = run(meta, path,
        dataset=loc, threshold=0.1, use_dask=use_dask, subset=subset,
        output_folder="/home/janrei1@ad.cmm.se/Desktop/22A5x4-2.subtr.reconstr.res/"
        )

    print("Calculating features")

    time_map_path = "/home/janrei1@ad.cmm.se/Desktop/22A5x4-2.subtr.reconstr.res/time_map.npy"
    event_map_path = "/home/janrei1@ad.cmm.se/Desktop/22A5x4-2.subtr.reconstr.res/event_map/"
    custom_slim_features(time_map_path, path, event_map_path)

    print("Saving")
    output_folder="/home/janrei1@ad.cmm.se/Desktop/22A5x4-2.subtr.reconstr.res/"
    if output_folder is not None:
        with open(f"{output_folder}/meta.json", 'w') as outfile:
            json.dump(meta, outfile)

    dt = time.time() - t0
    print("{:.1f} min".format(dt / 60) if dt > 60 else "{:.1f} s".format(dt))
