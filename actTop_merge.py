import argparse
import os
import random
import shutil

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
from skimage.filters import threshold_triangle, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import dask.array as da
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler
from dask_image import ndmorph, ndfilters, ndmeasure
from dask.distributed import Client, LocalCluster

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import multiprocessing as mp
# from multiprocessing import shared_memory

# import multiprocess as mp
from multiprocessing import cpu_count
from multiprocess import shared_memory
from itertools import repeat

import sys
import traceback
import warnings

import dask

import json

from pathlib import Path

import tifffile as tf
from dask.distributed import progress


# from pathos.multiprocessing import ProcessingPool as pPool

def analyze(event, arr):
    arr[event.label] = event.label


class EventDetector:

    # TODO decide on private and public funtions
    # TODO improve verbosity
    # TODO shared memory
    # TODO multicore
    # TODO define inactive pixels through rolling std deviation and use as mask
    # TODO use inactive pixels to estimate photo bleaching

    def __init__(self, input_path: str,
                 output: str = None, indices: np.array = None, verbosity: int = 1):

        # initialize verbosity
        self.vprint = self.verbosity_print(verbosity)
        self.last_timestamp = time.time()

        # paths
        self.input_path = Path(input_path)
        self.output = output
        working_directory = self.input_path.parent

        self.vprint(f"working directory: {working_directory}", 1)
        self.vprint(f"input file: {self.input_path}", 1)

        # TODO enforce tileDB input

        # TODO double check all of this
        # quality check arguments
        assert os.path.isfile(input_path) or os.path.isdir(input_path), f"input file does not exist: {input_path}"
        # assert output is None or ~ os.path.isfile(output), f"output file already exists: {output}"  # TODO assert output
        assert indices is None or indices.shape == (3, 2), "indices must be np.arry of shape (3, 2) -> ((z0, z1), " \
                                                           "(x0, x1), (y0, y1)). Found: " + indices

        # shared variables
        self.file = None
        self.Z, self.X, self.Y = None, None, None
        self.meta = {}

    def run(self, dataset=None,
            threshold=None, min_size=20, use_dask=False, adjust_for_noise=False,
            subset=None, output_folder=None):

        self.meta["subset"] = subset
        self.meta["threshold"] = threshold
        self.meta["min_size"] = min_size
        self.meta["adjust_for_noise"] = adjust_for_noise

        # output folder
        if self.output is None:
            # self.output_directory = self.input_path.with_suffix(".roi") if dataset is None else self.input_path.with_suffix(".roi_{}".format(dataset.split("/")[-1]))
            self.output_directory = self.input_path.with_suffix(
                ".roi") if dataset is None else self.input_path.with_suffix(".{}.roi".format(dataset.split("/")[-1]))
        else:
            self.output_directory = self.output

        if not self.output_directory.is_dir():
            os.mkdir(self.output_directory)
        self.vprint(f"output directory: {self.output_directory}", 1)

        # profiling
        pbar = ProgressBar(minimum=10)
        pbar.register()

        # TODO save this information somewhere
        # resources = ResourceProfiler()
        # resources.register()

        # load data
        data = self._load(dataset_name=dataset, use_dask=use_dask, subset=subset)
        self.data = data
        self.Z, self.X, self.Y = data.shape
        self.vprint(data if use_dask else data.shape, 2)

        # calculate event map
        event_map_path = self.output_directory.joinpath("event_map.tdb")
        if not os.path.isdir(event_map_path):
            self.vprint("Estimating noise", 2)
            # TODO maybe should be adjusted since it might already be calculated
            noise = self.estimate_background(data) if adjust_for_noise else 1

            self.vprint("Thresholding events", 2)
            event_map = self.get_events(data, roi_threshold=threshold, var_estimate=noise, min_roi_size=min_size)

            self.vprint(f"Saving event map to: {event_map_path}", 2)
            event_map.rechunk((100, 100, 100)).to_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            self.vprint(f"Saving tiff to : {tiff_path}", 2)
            tf.imwrite(tiff_path, event_map, dtype=event_map.dtype)

        else:
            self.vprint(f"Loading event map from: {event_map_path}", 2)
            event_map = da.from_tiledb(event_map_path.as_posix())

            tiff_path = event_map_path.with_suffix(".tiff")
            if not tiff_path.is_file():
                self.vprint(f"Saving tiff to : {tiff_path}", 2)
                tf.imwrite(tiff_path, event_map, dtype=event_map.dtype)

        # calculate time map
        self.vprint("Calculating time map", 2)
        time_map_path = self.output_directory.joinpath("time_map.npy")
        if not time_map_path.is_file():
            time_map = self.get_time_map(event_map)

            self.vprint(f"Saving event map to: {time_map_path}", 2)
            np.save(time_map_path, time_map)
        else:
            self.vprint(f"Loading time map from: {time_map_path}", 2)
            time_map = np.load(time_map_path.as_posix())

        # calculate features
        self.vprint("Calculating features", 2)
        self.custom_slim_features(time_map, self.input_path, event_map_path)

        self.vprint("saving features", 2)
        with open(self.output_directory.joinpath("meta.json"), 'w') as outfile:
            json.dump(self.meta, outfile)

        self.vprint("Run complete!", 1)

        # TODO do we really not want to calculate this?
        # return event_map, raw_trace_store, mask_store, footprints, meta

        # features = self.calculate_event_features(data, event_map, event_properties, 1, moving_average, threshold)
        # features = self.calculate_event_propagation(data, event_properties, features)

    def verbosity_print(self, verbosity_level: int = 5):

        """ Creates print function that takes global verbosity level into account """

        def v_level_print(msg: str, urgency: int):
            """ print function that checks if urgency is higher than global verbosity setting

            :param msg: message of the printout
            :param urgency: urgency level of the message
            :return:
            """

            if urgency <= verbosity_level:
                print("{}{} {:.2f}s {}".format("\t" * (urgency - 1), "*" * urgency, time.time() - self.last_timestamp,
                                               msg))

        return v_level_print

    def _load(self, dataset_name: str = None, use_dask: bool = False, subset=None):

        """ loads data from file

        :param dataset_name: name of the dataset in stored in an hdf file
        :param in_memory: flag whether to load full dataset in memory or not
        :return:
        """

        # TODO instead of self reference; prob better to explicitly give path as argument
        if self.input_path.suffix == ".h5":

            assert dataset_name is not None, "'dataset_name' required if providing an hdf file"

            file = h5.File(self.input_path, "r")
            assert dataset_name in file, "dataset '{}' does not exist in file".format(dataset_name)

            data = da.from_array(file[dataset_name], chunks='auto') if use_dask else file[dataset_name]

        elif self.input_path.suffix in (".tdb", ".delta"):

            data = tiledb.open(self.input_path.as_posix())

            if use_dask:
                data = da.from_array(data, chunks='auto')

        else:
            self.vprint(f"unknown file type: {self.input_path}")

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

    def get_events(self, data: np.array, roi_threshold: float, var_estimate: float,
                   min_roi_size: int = 10, mask_xy: np.array = None, smoXY=2,
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

        if roi_threshold is not None:
            active_pixels = da.from_array(np.zeros(data.shape, dtype=np.bool8))
            # self.vprint("Noise threshold: {:.2f}".format(roi_threshold * np.sqrt(var_estimate)), 4)

            absolute_threshold = roi_threshold * np.sqrt(var_estimate) if var_estimate is not None else roi_threshold
            active_pixels[:] = ndfilters.gaussian_filter(data, smoXY) > absolute_threshold

        else:
            self.vprint("no threshold defined. Using skimage.threshold to define threshold dynamically ...", 3)

            def dynamic_threshold(img):

                smooth = gaussian(img, sigma=smoXY)
                thr = 1 if np.sum(img) == 0 else threshold_triangle(smooth)
                img_thr = smooth > thr
                img_thr = img_thr.astype(np.bool_)

                return img_thr

            data_rechunked = data.rechunk((1, -1, -1))
            active_pixels = data_rechunked.map_blocks(dynamic_threshold, dtype=np.bool_)

        self.vprint("identified active pixels", 3)

        # mask inactive pixels (accelerates subsequent computation)
        if mask_xy is not None:
            np.multiply(active_pixels, mask_xy, out=active_pixels)
            self.vprint("masked inactive pixels", 3)

        # subsequent analysis hard to parallelize; save in memory
        if remove_small_object_framewise:
            active_pixels = active_pixels.compute() if type(active_pixels) == da.core.Array else active_pixels

            # remove small objects
            for cz in range(active_pixels.shape[0]):
                active_pixels[cz, :, :] = morphology.remove_small_objects(active_pixels[cz, :, :],
                                                                          min_size=min_roi_size, connectivity=4)
            self.vprint("removed small objects", 3)

            # label connected pixels
            # event_map, num_events = measure.label(active_pixels, return_num=True)
            event_map = np.zeros(data.shape, dtype=np.uintc)
            event_map[:], num_events = ndimage.label(active_pixels)
            self.vprint("labelled connected pixel. #events: {}".format(num_events), 3)

        else:

            # remove small objects
            struct = ndimage.generate_binary_structure(3, 4)

            active_pixels = ndmorph.binary_opening(active_pixels, structure=struct)
            active_pixels = ndmorph.binary_closing(active_pixels, structure=struct)
            self.vprint("removed small objects", 3)

            # label connected pixels
            event_map = da.from_array(np.zeros(data.shape, dtype=np.uintc))
            event_map[:], num_events = ndimage.label(active_pixels)
            self.vprint("labelled connected pixel. #events: {}".format(num_events), 3)

        # characterize each event

        # event_properties = measure.regionprops(event_map, intensity_image=data, cache=True,
        #                                        extra_properties=[self.trace, self.footprint]
        #                                        )
        # self.vprint("events collected", 3)

        if num_events < 2 * 32767:
            event_map = event_map.astype("uint16")
        else:
            event_map = event_map.astype("uint32")

        self.vprint("event_map dtype: {}".format(event_map.dtype), 4)

        return event_map  # , event_properties

    def calculate_event_features(self, data: np.array, event_map: np.array, event_properties: dict,
                                 seconds_per_frame: float, smoothing_window: int, prominence: float,
                                 background_fluorescence=0, use_poissoin_noise_model=False,
                                 skip_single_frame_events=True):

        """ calculate basic features of events

        :param data: 3D array of fluorescence signal
        :param event_map: 3D array of events labelled with event identifier
        :param event_properties: dictionary of event properties
        :param seconds_per_frame: time per frame
        :param smoothing_window: window for moving average smoothing
        :param prominence: prominence for detection of events in fluorescence trace
        :param background_fluorescence: offset background fluorescence
        :param use_poissoin_noise_model:
        :param skip_single_frame_events: skip events with only a single event frame
        :return: dict of with feature list (events, curves)
        """

        # TODO rename variable names

        Z, X, Y = data.shape
        if use_poissoin_noise_model:
            data = np.sqrt(data)

        # # replace events with median of surrounding (impute)
        # datx = data.copy()  # TODO really necessary?
        # print(datx.dtype)
        # datx[event_map > 0] = np.nan  # set all detected events None
        # for evt in event_properties:
        #
        #     # get bounding box
        #     bbox_z0, bbox_x0, bbox_y0, bbox_z1, bbox_x1, bbox_y1 = evt.bbox  # bounding box coordinates
        #     dInst = data[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1]  # bounding box intensity
        #     mskST = evt.image  # binary mask of active region
        #
        #     # calculate not NaN median for masks that are not 100% of the bounding box
        #     if evt.extent < 1:
        #         median_inv_mask = np.nanmedian(dInst[~mskST])
        #     else:
        #         median_inv_mask = np.min(dInst)
        #
        #     # replace NaN with median
        #     box = datx[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1]
        #     box[np.isnan(box)] = median_inv_mask
        #     datx[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1] = box

        Tww = min(smoothing_window, Z / 4)
        bbm = 0

        feature_list = {}

        # dMat = np.zeros((len(arLst_properties), Z, 2), dtype=np.single)
        # dffMat = np.zeros((len(arLst_properties), Z, 2), dtype=np.single)

        for i, evt in enumerate(event_properties):
            bbox_z0, bbox_x0, bbox_y0, bbox_z1, bbox_x1, bbox_y1 = evt.bbox  # bounding box coordinates

            # skip if event is only one frame long
            if skip_single_frame_events and bbox_z1 - bbox_z0 < 2:
                self.vprint("skipping event since it exists for less than 2 frames [{}]".format(i), 4)
                continue

            mskST = evt.image  # binary mask of active region
            mskSTSeed = evt.intensity_image  # real values of region (equivalent to dInst .* mskST)
            dInst = data[:, bbox_x0:bbox_x1, bbox_y0:bbox_y1]  # bounding box intensity; full length Z
            event_map_sel = event_map[:, bbox_x0:bbox_x1, bbox_y0:bbox_y1]
            xy_footprint = np.max(mskST, axis=0)  # calculate active pixels in XY

            # convert to numpy # TODO avoidable?
            dInst, xy_footprint, event_map_sel = dask.compute(dInst, xy_footprint, event_map_sel)

            dInst_x = dInst.copy()
            dInst_x[event_map_sel > 0] = None  # replace all events with None
            np.nan_to_num(dInst_x, copy=False, nan=np.nanmedian(dInst_x))
            # TODO fill NaN with sth more sensible (eg. surrounding non-none pixels or interpolate)

            # grab all frames with active pixels in the footprint
            active_frames = np.sum(event_map_sel, axis=(1, 2), where=xy_footprint)
            active_frames = active_frames > 0

            # > dFF
            raw_curve = np.mean(dInst, axis=(1, 2), where=xy_footprint)

            charx1 = self.detrend_curve(raw_curve, exclude=active_frames)

            sigma1 = np.sqrt(np.median(np.power(charx1[1:] - charx1[:-1], 2)) / 0.9113)

            charxBg1 = np.min(self.moving_average(charx1, Tww))
            charxBg1 -= bbm * sigma1  # TODO bbm is set to zero!?
            charxBg1 -= background_fluorescence

            dff1 = (charx1 - charxBg1) / charxBg1  # TODO this is terrible
            sigma1dff = np.sqrt(np.median(np.power(dff1[1:] - dff1[:-1], 2)) / 0.9113)

            dff1Sel = dff1[bbox_z0:bbox_z1]
            dff1Max = np.max(dff1Sel)

            # > dFF without other events
            raw_curve_noEvents = np.mean(dInst_x, axis=(1, 2), where=xy_footprint)
            raw_curve_noEvents[bbox_z0:bbox_z1] = raw_curve[bbox_z0:bbox_z1]  # splice current event back in

            current_event_frames = np.zeros(Z, dtype=np.bool_)
            current_event_frames[bbox_z0:bbox_z1] = 1
            charx1_noEvents = self.detrend_curve(raw_curve_noEvents, exclude=current_event_frames)
            sigma1_noEvents = np.sqrt(np.median(np.power(charx1_noEvents[1:] - charx1_noEvents[:-1], 2)) / 0.9113)

            charxBg1_noEvents = np.min(self.moving_average(charx1_noEvents, Tww))
            charxBg1_noEvents -= bbm * sigma1_noEvents  # TODO bbm is set to zero!?
            charxBg1_noEvents -= background_fluorescence

            dff1_noEvents = (charx1_noEvents - charxBg1_noEvents) / charxBg1_noEvents  # TODO this is terrible
            sigma1dff_noEvents = np.sqrt(np.median(np.power(dff1_noEvents[1:] - dff1_noEvents[:-1], 2)) / 0.9113)

            dff1Sel_noEvents = dff1_noEvents[bbox_z0:bbox_z1]

            # p_values
            dff1Max_noEvents = None
            if len(dff1Sel_noEvents) > 1:
                dff1Max_noEvents = np.max(dff1Sel_noEvents)
                dff_noEvents_tmax = np.argmax(dff1Sel_noEvents)

                xMinPre = max(np.min(dff1Sel_noEvents[:max(dff_noEvents_tmax, 1)]), sigma1dff)
                xMinPost = max(np.min(dff1Sel_noEvents[dff_noEvents_tmax:]), sigma1dff)
                dffMaxZ = np.max((dff1Max_noEvents - xMinPre + dff1Max_noEvents - xMinPost) / sigma1dff / 2, 0)
                dffMaxPval = 1 - norm.cdf(dffMaxZ)
            else:
                dffMaxZ = None
                dffMaxPval = None

            # > extend event window in the curve
            evtMap_ = event_map[:, bbox_x0:bbox_x1, bbox_y0:bbox_y1] > 0  # TODO THIS IS INSANITY
            evtMap_[evtMap_ == evt.label] = 0  # exclude current event
            bbox_z_ext, dff_ext = self.extend_event_curve(dff1_noEvents, evtMap_, (bbox_z0, bbox_z1))

            # > calculate curve statistics
            curve_stats = self._get_curve_statistics(dff_ext, seconds_per_frame, prominence, curve_label=i)

            # > save curve parameters

            # dffMat[i, :, 0] = dff1
            # dffMat[i, :, 1] = dff1_noEvents
            #
            # dMat[i, :, 0] = charx1
            # dMat[i, :, 1] = charx1_noEvents

            feature_list[i] = {}

            feature_list[i]["stats"] = curve_stats

            feature_list[i]["event"] = {
                "bbox_z": (bbox_z0, bbox_z1),
                "bbox_x": (bbox_x0, bbox_x1),
                "bbox_y": (bbox_y0, bbox_y1),
                "footprint": xy_footprint,
                "mask": evt.image,
                "label": evt.label,
                "area": evt.area,
                # "centroid": evt.centroid,
                # "inertia_tensor": evt.inertia_tensor,
                # "solidity": evt.solidity,
            }

            feature_list[i]["curve"] = {
                "rgt1": (bbox_z0, bbox_z1),
                "dff_max": dff1Max,
                "dff_noEvents_max": dff1Max_noEvents,
                "dff_max_z": dffMaxZ,
                "dff_max_pval": dffMaxPval,
                "duration": (bbox_z1 - bbox_z0) * seconds_per_frame,
                "AUC_raw": np.sum(raw_curve[bbox_z0:bbox_z1]),
                "AUC_dff": np.sum(dff1Sel_noEvents),
                "dFF": dff1,
                "dFF_noEvents": dff1_noEvents,
                "raw_curve": charx1,
                "raw_curve_noEvents": charx1_noEvents
            }

        return feature_list

    def calculate_slim_features(self, event_properties):

        num_events = len(event_properties)
        bboxes = np.array(
            [[p.bbox[0], p.bbox[3], p.bbox[1], p.bbox[4], p.bbox[2], p.bbox[5]] for p in event_properties])

        bbox_dim = np.array(
            [bboxes[:, 1] - bboxes[:, 0],  # z lengths
             bboxes[:, 3] - bboxes[:, 2],  # x lengths
             bboxes[:, 5] - bboxes[:, 4]]  # y lengths
        )

        """ Storage strategy
            The results of the feature extraction grow quickly in size with the number
            of frames and events. However, most events only occur in a tiny spatial
            and temporal footprint. Saving the results in ragged arrays is therefore
            preferable due to dense storage. The downside is that we need to save two
            separate pieces of information: the dense data array and an array containing
            the indices for each event.
        """

        # create ragged container arrays
        raw_trace_ind = np.cumsum(bbox_dim[0, :])  # indices for each event trace
        # raw_trace_store = da.from_array(np.memmap(tempfile.NamedTemporaryFile(), mode="w+",
        #                                           shape=raw_trace_ind[-1], dtype=np.single))
        raw_trace_store = da.from_array(np.zeros(shape=raw_trace_ind[-1], dtype=np.single))

        mask_ind = np.cumsum(np.prod(bbox_dim, axis=0))  # indices for each mask
        # mask_store = da.from_array(np.memmap(tempfile.NamedTemporaryFile(), mode="w+",
        #                                      shape=mask_ind[-1], dtype=np.byte))
        mask_store = da.from_array(np.zeros(shape=mask_ind[-1], dtype=np.byte))

        # footprints = da.from_array(np.memmap(tempfile.NamedTemporaryFile(), mode="w+",
        #                                      shape=(num_events, self.X, self.Y), dtype=np.byte))
        # footprints = da.from_array(np.zeros(shape=(num_events, self.X, self.Y), dtype=np.byte))

        # storage for smaller data
        meta_keys = ["raw_trace_ind_0", "raw_trace_ind_1", "mask_ind_0", "mask_ind_1",
                     "label", "area",  # "centroid", "orientation", "eccentricity",
                     "bbox_z0", "bbox_z1", "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1"]

        meta_lookup_tbl = {key: i for i, key in enumerate(meta_keys)}
        meta = da.from_array(np.zeros((num_events, len(meta_keys)), dtype=np.int32))

        # meta = {key: [] for key in
        #         ["raw_trace_ind_0", "raw_trace_ind_1", "mask_ind_0", "mask_ind_1",
        #          "label", "area",  # "centroid", "orientation", "eccentricity",
        #          "bbox_z0", "bbox_z1", "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1"]
        #         }

        # fill ragged containers
        for event_i, event_prop in tqdm(enumerate(event_properties), total=len(event_properties)):
            meta[event_i, meta_lookup_tbl["label"]] = event_prop.label
            intensity_image = event_prop.intensity_image
            mask = event_prop.image

            # bounding box
            z0, z1, x0, x1, y0, y1 = bboxes[event_i]
            meta[event_i, meta_lookup_tbl["bbox_z0"]] = z0
            meta[event_i, meta_lookup_tbl["bbox_z1"]] = z1
            meta[event_i, meta_lookup_tbl["bbox_x0"]] = x0
            meta[event_i, meta_lookup_tbl["bbox_x1"]] = x1
            meta[event_i, meta_lookup_tbl["bbox_y0"]] = y0
            meta[event_i, meta_lookup_tbl["bbox_y1"]] = y1

            # curve ROI only
            ind_0 = raw_trace_ind[event_i - 1] if event_i > 0 else 0
            ind_1 = raw_trace_ind[event_i]
            masked = da.ma.masked_array(intensity_image, np.invert(mask))
            raw_trace_store[ind_0:ind_1] = da.ma.filled(np.nanmean(masked, axis=(1, 2)))

            meta[event_i, meta_lookup_tbl["raw_trace_ind_0"]] = ind_0
            meta[event_i, meta_lookup_tbl["raw_trace_ind_1"]] = ind_1

            # boolean event mask
            ind_0 = mask_ind[event_i - 1] if event_i > 0 else 0
            ind_1 = mask_ind[event_i]
            mask_store[ind_0:ind_1] = mask.flatten()

            meta[event_i, meta_lookup_tbl["mask_ind_0"]] = ind_0
            meta[event_i, meta_lookup_tbl["mask_ind_1"]] = ind_1

            # save footprint
            # footprints[event_i, :, :] = np.zeros((self.X, self.Y))
            # footprints[event_i, x0:x1, y0:y1] = np.min(mask, axis=0)

            # small properties
            meta[event_i, meta_lookup_tbl["area"]] = event_prop.area
            # meta["centroid"].append(event_prop.centroid)
            # meta["orientation"].append(event_prop.orientation)  # not implemented for 3D
            # meta["eccentricity"].append(event_prop.eccentricity)  # not implemented for 3D

        return meta, meta_lookup_tbl, raw_trace_store, mask_store, None  # , footprints

    def save_slim_features(self, event_properties, output_folder):

        num_events = len(event_properties)
        bboxes = np.array(
            [[p.bbox[0], p.bbox[3], p.bbox[1], p.bbox[4], p.bbox[2], p.bbox[5]] for p in event_properties])

        bbox_dim = np.array(
            [bboxes[:, 1] - bboxes[:, 0],  # z lengths
             bboxes[:, 3] - bboxes[:, 2],  # x lengths
             bboxes[:, 5] - bboxes[:, 4]]  # y lengths
        )

        """ Storage strategy
            The results of the feature extraction grow quickly in size with the number
            of frames and events. However, most events only occur in a tiny spatial
            and temporal footprint. Saving the results in ragged arrays is therefore
            preferable due to dense storage. The downside is that we need to save two
            separate pieces of information: the dense data array and an array containing
            the indices for each event.
        """

        # create ragged container arrays
        raw_trace_ind = np.cumsum(bbox_dim[0, :])  # indices for each event trace
        self.meta["traces_indices"] = raw_trace_ind.tolist()
        memmap_trace = np.memmap(f"{output_folder}/trace.mmap", mode="w+", shape=raw_trace_ind[-1], dtype=np.single)
        raw_trace_store = da.from_array(memmap_trace)

        mask_ind = np.cumsum(np.prod(bbox_dim, axis=0))  # indices for each mask
        self.meta["mask_indices"] = mask_ind.tolist()
        memmap_mask = np.memmap(f"{output_folder}/mask.mmap", mode="w+", shape=mask_ind[-1], dtype=np.byte)
        mask_store = da.from_array(memmap_mask)

        # memmap_footprint = np.memmap(f"{output_folder}/footprint.mmap",
        #                              mode="w+", shape=(num_events, self.X, self.Y), dtype=np.byte)
        # footprints = da.from_array(memmap_footprint)

        ft_path = f"{output_folder}/footprint.tdb"
        footprints = da.zeros(shape=(num_events, self.X, self.Y), dtype=np.byte)
        footprints.to_tiledb(ft_path)
        footprints = footprints.store(tiledb.open(ft_path), compute=False, return_stored=True)

        # storage for smaller data
        summary_keys = ["raw_trace_ind_0", "raw_trace_ind_1", "mask_ind_0", "mask_ind_1", "label", "area", "bbox_z0",
                        "bbox_z1", "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1"]
        self.meta["summary_columns"] = summary_keys

        memmap_summary = np.memmap(f"{output_folder}/summary.mmap", mode="w+", shape=(num_events, len(summary_keys)),
                                   dtype=np.int32)
        summary = da.from_array(memmap_summary)

        # fill ragged containers
        for event_i, event_prop in tqdm(enumerate(event_properties), total=len(event_properties)):
            mask = event_prop.image

            # indices
            r_ind_0 = raw_trace_ind[event_i - 1] if event_i > 0 else 0
            r_ind_1 = raw_trace_ind[event_i]

            m_ind_0 = mask_ind[event_i - 1] if event_i > 0 else 0
            m_ind_1 = mask_ind[event_i]

            z0, z1, x0, x1, y0, y1 = bboxes[event_i]

            # bounding box
            summary[event_i, :] = [r_ind_0, r_ind_1, m_ind_0, m_ind_1,
                                   event_prop.label, event_prop.area,
                                   z0, z1, x0, x1, y0, y1]

            # curve ROI only
            raw_trace_store[r_ind_0:r_ind_1] = event_prop.trace

            # boolean event mask
            mask_store[m_ind_0:m_ind_1] = mask.flatten()

            # save footprint
            footprints[event_i, :, :] = np.zeros((self.X, self.Y))
            footprints[event_i, x0:x1, y0:y1] = np.min(mask, axis=0)

            # small properties
            # meta["centroid"].append(event_prop.centroid)

        return summary, raw_trace_store, mask_store, footprints

    # TODO anything useful in here?
    def custom_slim_features_legacy(self, event_map, parallel=True):

        # if parallel:

        # sh_data = shared_memory.SharedMemory(create=True, size=data.nbytes)
        # shn_data = np.ndarray(data.shape, dtype=data.dtype, buffer=sh_data.buf)
        # shn_data[:] = data

        sh_em = shared_memory.SharedMemory(create=True, size=event_map.nbytes)
        shn_em = np.ndarray(event_map.shape, dtype=event_map.dtype, buffer=sh_em.buf)
        shn_em[:] = event_map

        num_events = np.max(shn_em)

        with mp.Pool(mp.cpu_count()) as p:
            # TODO provide chunks; less overhead loading
            arguments = zip(range(1, num_events), repeat(event_map.shape), repeat(sh_em.name), repeat(self.file_path))
            num_task = num_events - 1

            R = []
            with tqdm(total=num_task) as pbar:
                for r in p.starmap(func, arguments):
                    R.append(r)
                    pbar.update()

        # sh_data.close()
        # sh_data.unlink()

        sh_em.close()
        sh_em.unlink()

        return R

    def custom_slim_features(self, time_map, data_path, event_path):

        # print(event_map)
        # sh_em = shared_memory.SharedMemory(create=True, size=event_map.nbytes)
        # shn_em = np.ndarray(event_map.shape, dtype=event_map.dtype, buffer=sh_em.buf)
        # shn_em[:] = event_map
        #
        # num_events = np.max(shn_em)

        # create chunks

        # shared memory
        self.vprint("creating shared memory arrays ...", 3)
        data = self.data
        n_bytes = data.shape[0] * data.shape[1] * data.shape[2] * data.dtype.itemsize  # get array info
        data_sh = shared_memory.SharedMemory(create=True, size=n_bytes)  # create shared buffer
        data_ = np.ndarray(data.shape, data.dtype, buffer=data_sh.buf)  # convert buffer to array
        data_[:] = data[:]  # load data
        data_info = (data.shape, data.dtype, data_sh.name)  # save info for use in task

        event = tiledb.open(event_path.as_posix())
        n_bytes = event.shape[0] * event.shape[1] * event.shape[2] * event.dtype.itemsize
        event_sh = shared_memory.SharedMemory(create=True, size=n_bytes)
        event_ = np.ndarray(event.shape, event.dtype, buffer=event_sh.buf)  # convert buffer to array
        event_[:] = event[:]  # load data
        event_info = (event.shape, event.dtype, event_sh.name)

        # collecting tasks
        self.vprint("collecting tasks ...", 3)

        e_start = np.argmax(time_map, axis=0)
        e_stop = time_map.shape[0] - np.argmax(time_map[::-1, :], axis=0)

        out_path = event_path.parent.joinpath("events/")
        if not out_path.is_dir():
            os.mkdir(out_path)

        # push tasks to client
        e_ids = list(range(1, len(e_start)))
        random.shuffle(e_ids)
        futures = []
        with Client(memory_limit='40GB') as client:

            for e_id in e_ids:
                futures.append(
                    client.submit(
                        characterize_event,
                        e_id, e_start[e_id], e_stop[e_id], data_info, event_info, out_path
                    )
                )
            progress(futures)

            client.gather(futures)

            # close shared memory
            try:
                data_sh.close()
                data_sh.unlink()

                event_sh.close()
                event_sh.unlink()
            except FileNotFoundError as err:
                print("An error occured during shared memory closing: ")
                print(err)

        # combine results
        events = {}
        for e in os.listdir(out_path):
            events.update(np.load(out_path.joinpath(e), allow_pickle=True)[()])
        np.save(event_path.parent.joinpath("events.npy"), events)
        shutil.rmtree(out_path)

        # else:

        # areas, traces, footprints = [], [], []
        # for i in tqdm(range(1, num_events)):
        #
        #     z, x, y = np.where(event_map == i)
        #     z0, z1 = np.min(z), np.max(z)
        #     x0, x1 = np.min(x), np.max(x)
        #     y0, y1 = np.min(y), np.max(y)
        #
        #     dz, dx, dy = z1-z0, x1-x0, y1-y0
        #     z, x, y = z-z0, x-x0, y-y0
        #
        #     mask = np.ones((dz+1, dx+1, dy+1), dtype=np.bool_)
        #     mask[(z, x, y)] = 0
        #
        #     signal = data[z0:z1+1, x0:x1+1, y0:y1+1]  # TODO weird that i need +1 here
        #     msignal = np.ma.masked_array(signal, mask)
        #
        #     # get number of pixels
        #     areas.append(len(z))
        #     traces.append(np.ma.filled(np.nanmean(msignal, axis=(1, 2))))
        #     footprints.append(np.min(mask, axis=0))
        #
        #     return areas, traces, footprints

    def get_time_map(self, event_map, chunk=200):

        time_map = np.zeros((event_map.shape[0], np.max(event_map) + 1), dtype=np.bool_)

        Z = event_map.shape[0]
        if type(event_map) == da.core.Array:

            for c in tqdm(range(0, Z, chunk)):

                cmax = min(Z, c + chunk)
                event_map_memory = event_map[c:cmax, :, :].compute()

                for z in range(c, cmax):
                    time_map[z, np.unique(event_map_memory[z - c, :, :])] = 1

        else:

            print("Assuming event_map is in RAM ... ")
            for z in tqdm(range(Z)):
                time_map[z, np.unique(event_map[z, :, :])] = 1

        return time_map

    def calculate_event_propagation(self, data, event_properties: dict, feature_list,
                                    spatial_resolution: float = 1, event_rec=None, north_x=0, north_y=1,
                                    propagation_threshold_min=0.2, propagation_threshold_max=0.8,
                                    propagation_threshold_step=0.1):

        # TODO this function is currently not working well dask. Fix!

        """

        :param data: 3D array of fluorescence data
        :param event_properties: dict of event properties (scipy.regionprops)
        :param feature_list: dict of event characteristics
        :param spatial_resolution: spatial resolution of each pixel in µm
        :param event_rec: legacy issue; probably best to leave at None
        :param north_x: bool dimension pointing north
        :param north_y: bool dimension pointing north
        :param propagation_threshold_min: minimum propagation threshold
        :param propagation_threshold_max: maximum propagation threshold
        :param propagation_threshold_step: propagation steps
        :return:
        """

        # in the original codebase additional steps were implemented
        # with current codebase evtRec will always be 255
        if event_rec is None:
            event_rec = 255 * np.ones(data.shape)

        feature_list["propagation_direction_order"] = ["north", "south", "west", "east"]
        kDi = [
            [north_x, north_y],
            [-north_x, -north_y],
            [-north_y, north_x],
            [north_y, -north_x]
        ]

        # define propagation thresholds
        if propagation_threshold_min == propagation_threshold_max:
            thr_range = [propagation_threshold_min]
        else:
            thr_range = np.arange(propagation_threshold_min, propagation_threshold_max, propagation_threshold_step)

        for i, evt in enumerate(event_properties):

            if i not in feature_list.keys():
                continue

            # get bounding box
            bbox_z0, bbox_x0, bbox_y0, bbox_z1, bbox_x1, bbox_y1 = evt.bbox  # bounding box coordinates
            dInst = data[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1]  # bounding box intensity
            mskST = evt.image  # binary mask of active region

            dInst, mskST = dask.compute(dInst, mskST)

            bbZ, bbX, bbY = mskST.shape

            """ currently not necessary until evtRec is implemented
            voli0 = mskST  # TODO stupid variable names
            volr0 = np.ones(voli0.shape)  # TODO would be something useful if complete AquA algorithm were implemented

            # prepare active pixels
            volr0[voli0 == 0] = 0  # TODO this is useless; could define volr0 differently
            volr0[volr0 < min(thr_range)] = 0  # exclude pixels below lowest threshold
            
            sigMapXY = np.sum(volr0 >= min(thr_range), axis=0)
            nPix = np.sum(sigMapXY > 0)
            """
            nPix = np.sum(mskST)

            # mask for directions relative to 1st frame centroid
            try:
                centroid_x0, centroid_y0 = np.round(measure.centroid(mskST[0, :, :])).astype(int)
            except:

                print(type(mskST))
                print(mskST)

                time.sleep(1)
                traceback.print_exc()

            msk = np.zeros((bbX, bbY, 4))
            msk[:centroid_x0, :, 0] = 1  # north
            msk[centroid_x0:, :, 1] = 1  # south
            msk[:, :centroid_y0, 2] = 1  # west
            msk[:, centroid_y0:, 3] = 1  # east

            # locations of centroid
            sigDist = np.full((bbZ, 4, len(thr_range)), np.nan)  # TODO could probably also be zeros
            boundary = {}
            propMaxSpeed = np.zeros((bbZ, len(thr_range)))
            pixNum = np.zeros((bbZ, len(thr_range)))
            for cz in range(bbZ):

                current_frame = mskST[cz, :, :]
                for thr_i, thr in enumerate(thr_range):

                    # threshold current frame
                    current_frame_thr = current_frame >= thr
                    pixNum[cz, thr_i] = np.sum(current_frame_thr)

                    # define boundary / contour
                    current_frame_thr_closed = morphology.binary_closing(current_frame_thr)  # fill holes
                    dim_x, dim_y = current_frame_thr_closed.shape
                    if dim_x < 3 or dim_y < 3:
                        continue

                    cur_contour = measure.find_contours(current_frame_thr_closed)

                    if len(cur_contour) < 1:
                        continue
                    cur_contour = np.array(cur_contour[0])[:, ::-1]
                    boundary[(cz, thr_i)] = cur_contour

                    # calculate propagation speed
                    if cz > 1:

                        prev_contour = boundary[(cz - 1, thr_i)]
                        for px in cur_contour:
                            shift = px - prev_contour
                            dist = np.sqrt(np.power(shift[:, 0], 2) + np.power(shift[:, 1], 2))
                            current_speed = min(dist)
                            propMaxSpeed[cz, thr_i] = max(propMaxSpeed[cz, thr_i], current_speed)

                        for px in prev_contour:
                            shift = px - cur_contour
                            dist = np.sqrt(np.power(shift[:, 0], 2) + np.power(shift[:, 1], 2))
                            current_speed = min(dist)
                            propMaxSpeed[cz, thr_i] = max(propMaxSpeed[cz, thr_i], current_speed)

                    # mask with directions
                    for direction_i in range(4):
                        img0 = np.multiply(current_frame_thr, msk[:, :, direction_i])
                        img0_mask = img0 > 0

                        curr_centroid_x0, curr_centroid_y0 = measure.centroid(img0_mask[:, :])
                        dx, dy = curr_centroid_x0 - centroid_x0, curr_centroid_y0 - centroid_y0

                        sigDist[cz, direction_i, thr_i] = sum(np.multiply([dx, dy], kDi[direction_i]))

            # collect results
            prop = np.zeros(sigDist.shape) - 1  # np.full(sigDist.shape, None)
            prop[1:, :, :] = sigDist[1:, :, :] - sigDist[:-1, :, :]
            # prop[1:, :, :] = np.nansum(sigDist[1:, :, :], -1*sigDist[:-1, :, :])

            propGrowMultiThr = prop.copy()
            propGrowMultiThr[propGrowMultiThr < 0] = None
            propGrow = np.nanmax(propGrowMultiThr, axis=2)
            propGrow[np.isnan(propGrow)] = 0
            propGrowOverall = np.nansum(propGrow, 0)

            propShrinkMultiThr = prop.copy()
            propShrinkMultiThr[propShrinkMultiThr > 0] = None
            propShrink = np.nanmax(propShrinkMultiThr, axis=2)
            propShrink[np.isnan(propShrink)] = 0
            propShrinkOverall = np.nansum(propShrink, 0)

            pixNumChange = np.zeros(pixNum.shape)
            pixNumChange[1:] = pixNum[1:, :] - pixNum[:-1, :]
            pixNumChangeRateMultiThr = pixNumChange / nPix
            pixNumChangeRateMultiThrAbs = np.abs(pixNumChangeRateMultiThr)
            pixNumChangeRate = np.max(pixNumChangeRateMultiThrAbs, axis=1)

            # save results
            if i not in feature_list.keys():
                self.vprint("ROI features doesn't exist [{}]".format(i), 4)
                feature_list[i] = {}

            feature_list[i]["propagation"] = {
                "propGrow": propGrow * spatial_resolution,
                "propGrowOverall": propGrowOverall * spatial_resolution,
                "propShrink": propShrink * spatial_resolution,
                "propShrinkOverall": propShrinkOverall * spatial_resolution,
                "areaChange": pixNumChange * spatial_resolution * spatial_resolution,
                "areaChangeRate": pixNumChangeRate,
                "areaFrame": pixNum * spatial_resolution * spatial_resolution,
                "propMaxSpeed": propMaxSpeed * spatial_resolution,
                "maxPropSpeed": np.max(propMaxSpeed),
                "avgPropSpeed": np.mean(propMaxSpeed)
            }

        return feature_list

    def _get_curve_statistics(self, curve: np.array, seconds_per_frame: float, prominence: float,
                              curve_label: str = None, relative_height: list = (0.1, 0.5, 0.9),
                              enforce_single_peak: bool = True, max_iterations: int = 100,
                              ignore_tau=False, min_fit_length=5):

        """ calculate characteristic of fluorescence trace

        :param curve: 1D array of fluorescence values
        :param seconds_per_frame: time per frame
        :param prominence: prominence of events in curve (see scipy.prominence)
        :param curve_label: (optional) identifier for curve; used for exception handling
        :param relative_height: list of heights at which curve stats are calculated
        :param enforce_single_peak: flag decided whether prominence is iteratively
                adjusted until only one event is found in curve
        :param max_iterations: maximum iterations to enforce single event detection
        :param ignore_tau: flag to skip exponential fitting step
        :return:
        """

        curve_stat = {}

        # > identify peak
        peak_x, peak_props = signal.find_peaks(curve, prominence=prominence)

        num_iterations = 0
        while enforce_single_peak and num_iterations <= max_iterations:

            # correct number of peaks
            if len(peak_x) == 1:
                break

            # not enough peaks found
            if len(peak_x) < 1:
                prominence *= 0.9

            # too many peaks found
            if len(peak_x) > 1:
                prominence *= 1.05

            peak_x, peak_props = signal.find_peaks(curve, prominence=prominence)
            num_iterations += 1

        if len(peak_x) < 1:
            self.vprint("Warning: no peak identified in curve [#{}]. Consider increasing 'max_iterations' "
                        "or improving baseline subtraction".format(len(peak_x)), 4)
            curve_stat = {}
            return curve_stat

        if len(peak_x) > 1:
            self.vprint("Warning: more than one peak identified in curve [#{}]. Consider increasing 'max_iterations' "
                        "or improving baseline subtraction".format(len(peak_x)), 4)
            peak_x = [peak_x[0]]

        peak_x = peak_x[0]

        peak_prominence = peak_props["prominences"][0]
        peak_right_base = peak_props["right_bases"][0]
        peak_left_base = peak_props["left_bases"][0]

        curve_stat["x"] = peak_x
        curve_stat["prominence"] = peak_prominence
        curve_stat["height"] = curve[peak_x]
        curve_stat["left_base"] = peak_left_base
        curve_stat["right_base"] = peak_right_base
        curve_stat["rise_time"] = (peak_x - peak_left_base) * seconds_per_frame
        curve_stat["fall_time"] = (peak_right_base - peak_x) * seconds_per_frame
        curve_stat["width"] = (peak_right_base - peak_left_base) * seconds_per_frame

        # > identify relative width
        for nThr, thr in enumerate(relative_height):
            # find position at which curve crosses threshold closest to peak
            peak_char = signal.peak_widths(curve, [peak_x], rel_height=thr)
            peak_width, peak_width_height, peak_left_ips, peak_right_ips = peak_char

            curve_stat["peak_width_{}".format(thr * 100)] = peak_width
            curve_stat["peak_height_{}".format(thr * 100)] = peak_width_height
            curve_stat["peak_left_ips_{}".format(thr * 100)] = peak_left_ips
            curve_stat["peak_right_ips_{}".format(thr * 100)] = peak_right_ips

            curve_stat["rise_{}".format(thr * 100)] = (curve_stat["x"] - peak_left_ips) * seconds_per_frame
            curve_stat["fall_{}".format(thr * 100)] = (peak_right_ips - curve_stat["x"]) * seconds_per_frame
            curve_stat["width_{}".format(thr * 100)] = (peak_right_ips - peak_left_ips) * seconds_per_frame

        # > fit exponential decay
        if ~ ignore_tau or (peak_right_base - peak_x) < min_fit_length:
            Y = curve[peak_x:peak_right_base]
            X = range(0, peak_right_base - peak_x)

            def exp_func(x, a, b, c):
                return a * np.exp(-b * x) + c

            popt = None
            try:

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(exp_func, X, Y)

            except (RuntimeError, TypeError, ValueError) as err:
                self.vprint("Error occured in curve fitting for exponential decay. [#{}] {}".format(
                    curve_label if curve_label is not None else "?",
                    err
                ), 4)

            if popt is not None:
                curve_stat["decayTau"] = -1 / popt[1] * seconds_per_frame
            else:
                curve_stat["decayTau"] = None

        return curve_stat

    @staticmethod
    def trace(mask, intensity):
        masked = da.ma.masked_array(intensity, np.invert(mask))
        return da.ma.filled(np.nanmean(masked, axis=(1, 2)))

    @staticmethod
    def footprint(mask):
        return np.min(mask, axis=0)

    @staticmethod
    def get_start(arr, positions):
        return positions[0]

    @staticmethod
    def get_stop(arr, positions):
        return positions[-1]

    def subtract_background(self, data: np.array, trial_length, window, var_estimate=None, mask_xy=None):

        """ subtracts background signal by modelling video variance

        :param data: 3D array of fluorescence signal
        :param trial_length: trial length to calculate bias
        :param window: moving average window used for smoothing data
        :param var_estimate: estimated variance across the video
        :param mask_xy: binary 2D mask to ignore certain pixel
        :return: fluorescence signal after background subtraction
        """

        # TODO implement dynamic var_estimate

        # cut: frames per segment

        if var_estimate is None:
            var_estimate = self.estimate_background(data, mask_xy)

        Z, X, Y = data.shape

        dF = np.zeros(data.shape, dtype=np.single)

        # > calculate bias
        num_trials = 10000
        xx = np.random.randn(num_trials, trial_length) * var_estimate

        xxMA = np.zeros(xx.shape)
        for n in range(num_trials):
            xxMA[n, :] = self.moving_average(xx[n, :], window)

        xxMin = np.min(xxMA, axis=1)
        xBias = np.nanmean(xxMin)

        # > calculate dF
        for ix in range(X):
            for iy in range(Y):
                dF[:, ix, iy] = data[:, ix, iy] - min(self.moving_average(data[:, ix, iy], window)) - xBias

        return dF

    @staticmethod
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

    @staticmethod
    def moving_average(arr: np.array, window: int, axis: int = 0) -> np.array:

        """ fast implementation of windowed average

        :param arr: np.array of 1D, 2D or 3D shape. First dimension will be used for windowing
        :param window: window size for averaging
        :param axis: axis to reduce
        :param dask: flag to use dask multiprocessing
        :return: array of same size as input
        """

        if type(arr) == da.core.Array:

            depth = {0: window, 1: 0, 2: 0}

            overlapping_grid = da.overlap.overlap(arr, depth=depth, boundary='reflect')
            mov_average_grid = overlapping_grid.map_blocks(ndimage.filters.uniform_filter, window)
            trimmed_grid = da.overlap.trim_internal(mov_average_grid, depth)

            return trimmed_grid

        else:

            size = np.zeros(len(arr.shape))
            size[axis] = window

            return ndimage.filters.uniform_filter(arr, size=size)

    def detrend_curve(self, curve: np.array, exclude: np.array = None) -> np.array:

        """ detrends curve with first order polynomial

        :param curve: 1D array of fluorescence values
        :param exclude: 1D binary mask; 1 equals exclusion
        :return: detrended curve
        """

        # TODO this implementation makes very little sense with our data

        X = np.array(range(len(curve)))
        Y = curve

        # exclude activate frames from calculating baseline
        if exclude is not None:
            X = X[~exclude]
            Y = Y[~exclude]

        if len(Y) == 0:
            self.vprint("All data points were excluded due to active frames. Returning original Y instead", 4)
            return curve

        # define first order polynomial function
        def poly1(x, m, n):
            return m * x + n

        # fit baseline
        popt, pcov = curve_fit(poly1, X, Y)

        # calculate baseline for complete time range
        y_fit = poly1(range(len(curve)), *popt)

        # subtract baseline
        curve -= y_fit
        curve -= np.min(curve) + min(Y)

        return curve

    @staticmethod
    def extend_event_curve(curve, event_timings, bbox_z):

        """ extends event in time until it reaches prior/next event

        :param curve: 1D fluorescence trace
        :param event_timings: 1D binary mask where 1 equals existence of other events
        :param bbox_z: (z0, z1) binding box of event; z0 denoting start and z1 the end of event
        :return: new_bbox_z, extended curve
        """

        bbox_z0, bbox_z1 = bbox_z
        event_timings = np.min(event_timings, axis=(1, 2))

        # TODO check if this is actually legit
        #  reasoning: if there are no other events in the footprint
        #  then we can just analyze the full curve, no?
        if np.sum(event_timings) == 0:
            return (0, len(curve)), curve

        T = len(curve)
        t0 = max(bbox_z0 - 1, 0)
        t1 = min(bbox_z1 + 1, T - 1)

        # find local minimum between prior/next event and the current event
        try:
            if bbox_z0 < 1:  # bbox starts with first frame
                t0_min = bbox_z0
            elif np.sum(event_timings[:t0]) < 1:  # no other events prior to event
                t0_min = 0
            else:
                i0 = t0 - np.argmax(event_timings[:t0:]) - 1  # closest event prior to current
                t0_min = i0 + np.argmin(curve[i0:bbox_z0]) - 1

            if bbox_z1 > T - 1:  # bbox ends with last frame
                t1_min = bbox_z1
            elif np.sum(event_timings[t1:]) < 1:  # no other events post event
                t1_min = T
            else:
                i1 = np.argmax(event_timings[t1:]) + t1  # closest event post current event
                t1_min = bbox_z1 + np.argmin(curve[bbox_z1:i1]) - 1

        except ValueError as err:

            # print("bbox_z1:T > ", bbox_z1, T)
            # print("bbox_z1:i1 > ", bbox_z1, i0)
            # print(event_timings[t1:])

            print("i0:bbox_z0 > ", i0, bbox_z0)
            print("np.sum(event_timings[:t0]) > ", np.sum(event_timings[:t0]))
            print("np.argmax(event_timings[:t0:-1])", np.argmax(event_timings[:t0:-1]))
            print("event_timings[:t0:-1]", event_timings[:t0:-1])

            print("bbox_z1:i1 > ", bbox_z1, i1)

            time.sleep(1)
            print(err)
            traceback.print_exc()
            sys.exit(2)

        # ?
        if t0_min >= t1_min:
            t0_min = t0
            t1_min = t1

        return (t0_min, t1_min), curve[t0_min:t1_min]

    @staticmethod
    def get_local_maxima(data, event_map, event_properties, smoothing_kernel=(1, 1, 0.5), bounding_box_extension=3):

        """ Detect all local maxima in the video

        :param data: 3D of fluorescence data
        :param event_map: 3D array with labelled events (event identifier)
        :param event_properties: scipy.regionprops of detected events
        :param smoothing_kernel: kernel for gaussian smoothing
        :param bounding_box_extension: value by which the bounding box is extended in all dimensions
        :return:
        """

        Z, X, Y = data.shape

        # detect in active regions only
        # lmAll = np.zeros(data.shape, dtype=np.bool_)
        lmLoc = []  # location of local maxima
        lmVal = []  # values of local maxima
        for i, prop in enumerate(event_properties):

            # pix0 = prop.coords  # absolute coordinates
            bbox_z0, bbox_x0, bbox_y0, bbox_z1, bbox_x1, bbox_y1 = prop.bbox  # bounding box coordinates
            # pix0_relative = pix0 - (bbox_z0, bbox_x0, bbox_y0)  # relative coordinates

            # extend bounding box
            bbox_z0 = max(bbox_z0 - bounding_box_extension, 0)
            bbox_z1 = min(bbox_z1 + bounding_box_extension, Z)

            bbox_x0 = max(bbox_x0 - bounding_box_extension, 0)
            bbox_x1 = min(bbox_x1 + bounding_box_extension, X)

            bbox_y0 = max(bbox_y0 - bounding_box_extension, 0)
            bbox_y1 = min(bbox_y1 + bounding_box_extension, Y)

            # get mask and image intensity
            if bounding_box_extension < 1:
                mskST = prop.image  # binary mask of active region
                dInst = data[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1]  # bounding box intensity
                mskSTSeed = prop.intensity_image  # real values of region (equivalent to dInst .* mskST)
            else:
                mskST = event_map[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1] > 0
                dInst = data[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1]
                mskSTSeed = np.multiply(dInst, mskST)

            # get local maxima
            dInst_smooth = ndimage.gaussian_filter(dInst, sigma=smoothing_kernel)
            local_maxima = morphology.local_maxima(dInst_smooth, allow_borders=False, indices=True)
            lm_z, lm_x, lm_y = local_maxima

            # select local maxima within mask
            for lmax in zip(lm_z, lm_x, lm_y):

                if mskST[lmax] > 0:
                    lmLoc.append(lmax)
                    lmVal.append(mskSTSeed[lmax])

            return lmLoc, lmVal


def export_to_tdb(path, loc=None, out=None):
    if out is None:
        out = path + "_{}.tdb".format(loc)
        print("Output: ", out)

    if path.endswith(".h5"):
        assert loc is not None, "please provide a dataset name 'loc"

        with h5.File(path, "r") as file:
            data = file[loc]

            data = da.from_array(data)
            data.to_tiledb(out)


def print_array_size(dimension, dtype):
    A = np.zeros(dimension, dtype=dtype)
    print(
        "nBytes: {} ... {}GB ... max: {}-{}".format(A.nbytes, A.nbytes / 1e9, np.iinfo(dtype).min, np.iinfo(dtype).max))


def p(arr1, arr2, nf=None):
    if nf is None:
        nf = [0, int(arr1.shape[0] / 2), -1]

    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax0, ax1 = list(ax.flatten())

    spacer = np.ones((600, 1))

    im0 = [arr1[n, :, :] for n in nf]
    im1 = [arr2[n, :, :] for n in nf]

    ax0.imshow(np.concatenate(im0, axis=1))
    ax1.imshow(np.concatenate(im1, axis=1))

    plt.show()


class Legacy:
    ##############
    ## advanced ##
    ##############

    def arSimPrep(data, smoXY):
        mskSig = np.var(data, 0) > 1e-8

        dat = data.copy()  # TODO inplace possible?
        dat = dat + np.random.randn(*dat.shape) * 1e-6

        # smooth data
        if smoXY > 0:
            for z in range(dat.shape[0]):
                dat[z, :, :] = ndimage.gaussian_filter(dat[z, :, :], sigma=smoXY, truncate=np.ceil(2 * smoXY) / smoXY,
                                                       mode="wrap")

        # estimate noise
        stdEst = calc_noise(dat, mskSig)

        dF = getDfBlk(dat, cut=200, movAvgWin=25, stdEst=stdEst)  # TODO cut, movAvgWin variable

        return dat, dF, stdEst

    def getARSim(data, smoMax, thrMin, minSize, evtSpatialMask=None,
                 smoCorr_location="smoCorr.h5"):

        # TODO in this sSim is currently orders of magnitude too small compared to matlab implementation not clear
        #  why that is

        # learn noise correlation
        T, H, W = data.shape
        T1 = min(T, 100)
        datZ = zscore(data[:T1, :, :], 0)

        rhoX = np.mean(np.multiply(datZ[:, :-1, :], datZ[:, 1:, :]), axis=0)
        rhoY = np.mean(np.multiply(datZ[:, :, :-1], datZ[:, :, 1:]), axis=0)
        rhoXM = np.nanmedian(rhoX)
        rhoYM = np.nanmedian(rhoY)

        with h5.File(smoCorr_location) as smoCorr:
            cx = smoCorr['cx'][:]
            cy = smoCorr['cy'][:]
            sVec = smoCorr['sVec'][:]

        ix = np.argmin(abs(rhoXM - cx))
        iy = np.argmin(abs(rhoYM - cy))
        smo0 = sVec[max(ix, iy)][0]

        dSim = np.random.randn(200, H, W) * 0.2
        sigma = smo0
        dSim = ndimage.gaussian_filter(dSim, sigma=sigma,  # TODO supposed to be 2D, not 3D
                                       truncate=np.ceil(2 * sigma) / sigma, mode="wrap")

        rto = data.shape[0] / dSim.shape[0]

        # > simulation
        smoVec = [smoMax]  # TODO looks like this should be a vector!?!?!?!
        thrVec = np.arange(thrMin + 3, thrMin - 1, -1)

        t0 = time.time()
        dARAll = np.zeros((H, W))
        for i, smoI in enumerate(smoVec):

            print("Smo {} [{}] ==== - {:.2f}\n".format(i, smoI, (time.time() - t0) / 60))
            # opts.smoXY = smoI ## ???

            _, dFSim, sSim = arSimPrep(dSim, smoI)
            _, dFReal, sReal = arSimPrep(data, smoI)

            for j, thrJ in enumerate(thrVec):

                dAR = np.zeros((T, H, W))
                print("Thr{}:{} [> {:.2f}| >{:.2f}]  - {:.2f}\n".format(j, thrJ, thrJ * sSim, thrJ * sReal,
                                                                        (time.time() - t0) / 60))

                # > null
                print("\t calculating null ... - {:.2f}".format((time.time() - t0) / 60))
                tmpSim = dFSim > thrJ * sSim
                szFreqNull = np.zeros((H * W))
                for cz in range(T):  # TODO shouldn't be range T but 200!?
                    # tmp00 = np.multiply(tmpReal[cz, :, :], evtSpatialMask)
                    tmp00 = tmpSim[cz, :, :]

                    # identify distinct areas
                    labels = measure.label(tmp00)  # dim X.Y, int
                    areas = [prop.area for prop in measure.regionprops(labels)]

                    for area in areas:
                        szFreqNull[area] += 1

                szFreqNull = szFreqNull * rto

                # > observation
                print("\t calculating observation ... - {:.2f}".format((time.time() - t0) / 60))
                tmpReal = dFReal > thrJ * sReal
                szFreqObs = np.zeros(H * W)  # counts frequency of object size
                maxArea = 0  # used to speed up false positive control (see below)
                for cz in range(T):
                    # tmp00 = np.multiply(tmpReal[cz, :, :], evtSpatialMask)
                    tmp00 = tmpReal[cz, :, :]

                    labels = measure.label(tmp00)  # dim X.Y, int
                    areas = [prop.area for prop in measure.regionprops(labels)]
                    maxArea = max(maxArea, max(areas))

                    for area in areas:
                        szFreqObs[area] = szFreqObs[area] + 1

                # > false positive control: choose higher min_size threshold if null data suggests it
                # print("\t calculating false positives ... - {:.2f}".format((time.time()-t0)/60))
                # suc = 0
                # szThr = 0
                # for s in range(maxArea+1):
                #
                #     fpr = np.sum(szFreqNull[s:]) / np.sum(szFreqObs[s:])
                #
                #     if fpr < 0.01:
                #         suc = 1
                #         szThr = np.ceil(s*1.2)
                #         break
                #
                # szThr = max(szThr, minSize)
                szThr = minSize
                suc = 1
                print("\t\t szThr: {:.2f}".format(szThr))

                # > apply to data
                print("\t applying to data ... - {:.2f}".format((time.time() - t0) / 60))
                if suc > 0:

                    e00 = int(np.ceil(smoI / 2))
                    for cz in range(T):
                        tmp0 = tmpReal[cz, :, :]
                        tmp0 = morphology.remove_small_objects(tmp0, min_size=szThr)

                        if e00 > 0:
                            # erode image with square footprint
                            tmp0 = morphology.binary_erosion(tmp0, footprint=np.ones((2, e00)))

                        dAR[cz, :, :] = dAR[cz, :, :] + tmp0

                dARAll = dARAll + dAR

        dARAll = dARAll > 0
        arLst = measure.label(dARAll)
        arLst_properties = measure.regionprops(arLst)

        return dARAll, arLst, arLst_properties

    def getDfBlk_faster(path, loc, cut, movAvgWin, x_max=None, y_max=None, stdEst=None, spatialMask=None,
                        multiprocessing=True):
        # cut: frames per segment

        if stdEst is None:
            stdEst = calc_noise(data, spatialMask)

        with h5.File(path, "r") as file:
            data = file[loc]
            Z, X, Y = data.shape
            X = min(X, x_max)
            Y = min(Y, y_max)
            cz, cx, cy = data.chunks

        dF = np.zeros([Z, X, Y], dtype=np.single)  # TODO maybe double?

        # > calculate bias
        num_trials = 10000
        xx = np.random.randn(num_trials, cut) * stdEst

        xxMA = np.zeros(xx.shape)
        for n in range(num_trials):
            xxMA[n, :] = moving_average(xx[n, :], movAvgWin)

        xxMin = np.min(xxMA, axis=1)
        xBias = np.nanmean(xxMin)

        # > calculate dF
        if multiprocessing:

            # def subtract_baseline(trace0, movAvgWin, xBias):
            #     trace0 - min(moving_average(trace0, movAvgWin)) - xBias

            chunks = list(product(np.arange(0, X, cx), np.arange(0, Y, cy)))
            print("chunks {}: {}".format(len(chunks), chunks))
            dF_shared = shared_memory.SharedMemory(create=True, size=dF.nbytes)
            print("Shared Name: ", dF_shared.name)

            with mp.Pool(mp.cpu_count()) as pool:

                for x, y in chunks:
                    pool.apply(moving_average_h5, args=(path, loc, dF_shared.name, x, y, movAvgWin,
                                                        dF.shape, dF.dtype))

            dF[:] = dF_shared[:]

            dF_shared.close()
            dF_shared.unlink()

        else:
            for ix in range(X):
                for iy in range(Y):
                    dF[:, ix, iy] = data[:, ix, iy] - min(moving_average(data[:, ix, iy], movAvgWin)) - xBias

        return dF

    def func_actTop(self, raw, dff, foreground_threshold=0, in_memory=True,
                    noise_estimation_method="original"):

        assert noise_estimation_method in ["original"]

        Z, X, Y = raw.shape

        # > reserve memory
        mean_project = np.zeros((X, Y))  # reserve memory for mean projection
        var_project = np.zeros((X, Y))  # reserve memory for variance projection
        noise_map = np.zeros((X, Y))  # reserve memory for noise map

        # > Calculate mean projection
        if noise_estimation_method == "original":
            reference_frame = raw[-1, :, :]

        vprint("calculating projections ...", 1)
        if in_memory:
            mean_project[:] = np.mean(raw, 0)
            var_project[:] = np.var(raw, 0)
            noise_map[:] = self.calculate_noise(raw)
        else:
            # TODO PARALLEL

            cz, cx, cy = raw.chunks

            chunk = np.zeros((Z, cx, cy))
            # temp_chunk = np.zeros((Z-1, cx, cy))
            # temp_img = np.zeros((cx, cy))

            for ix in np.arange(0, X, cx):
                for iy in np.arange(0, Y, cy):
                    max_x = min(X - ix, cx)  # at most load till X border of image
                    max_y = min(Y - iy, cy)  # at most load till Y border of image

                    chunk[:, 0:max_x, 0:max_y] = raw[:, ix:ix + max_x, iy:iy + max_y]  # load chunk

                    mean_project[ix:ix + max_x, iy:iy + max_y] = np.mean(chunk, 0)
                    var_project[ix:ix + max_x, iy:iy + max_y] = np.var(chunk, 0)

                    noise_map[ix:ix + max_x, iy:iy + max_y] = self.calculate_noise(
                        chunk, reference_frame, chunked=True,
                        ix=ix, iy=iy, max_x=max_x, max_y=max_y
                    )

        # > Create masks
        msk000 = var_project > 1e-8  # non-zero variance # TODO better variable name

        msk_thrSignal = mean_project > foreground_threshold
        noiseEstMask = np.multiply(msk000, msk_thrSignal)  # TODO this might be absolutely useless for us

        # > noise level
        vprint("calculating noise ...", 1)
        sigma = 0.5
        noise_map_gauss_before = ndimage.gaussian_filter(noise_map, sigma=sigma,
                                                         truncate=np.ceil(2 * sigma) / sigma, mode="wrap")
        noise_map_gauss_before[noiseEstMask == 0] = None

        # > smooth data
        # TODO SMOOTH --> IO/memory intensive; possible to do without reloading and keeping copy in RAM?

        """
    
        temp_chunk[:, 0:max_x, 0:max_y] = np.power(
            #TODO why exclude the first frame
            np.subtract(chunk[1:, 0:max_x, 0:max_y], reference_frame[ix:ix+max_x, iy:iy+max_y]),
            2)  # dim: cz, cx, cy
    
        # dist_squared_median
        # TODO why 0.9133
        temp_img[0:max_x, 0:max_y] = np.median(temp_chunk[:, 0:max_x, 0:max_y], 0) / 0.9133  # dim: cx, cy
    
        # square_root
        noise_map[ix:ix+max_x, iy:iy+max_y] = np.power(temp_img[0:max_x, 0:max_y], 0.5)  # dim: cx, cy
    
    
        """

    ########
    ## h5 ##
    ########

    def moving_average_h5(path, loc, output, x, y, w,
                          shmem_shape, shmem_dtype):
        with h5.File(path, "r") as file:

            data = file[loc]
            cz, cx, cy = data.chunks

            chunk = data[:, x:x + cx, y:y + cy]
            Z, X, Y = chunk.shape

        buffer = mp.shared_memory.SharedMemory(name=output)
        dF = np.ndarray(shmem_shape, dtype=shmem_dtype, buffer=buffer.buf)

        for xi in range(X):
            for yi in range(Y):
                dF[:, x + xi, y + yi] = ndimage.filters.uniform_filter1d(chunk[:, xi, yi], size=w)

    def temp(path, x_range, y_range, thrARScl, varEst, minSize,
             buffer_name, buffer_dtype, buffer_shape,
             location="dff/neu", evtSpatialMask=None):
        print("temp")

        with h5.File(path, "r") as f:
            data = f[location][:, x_range[0]:x_range[1], y_range[0]:y_range[1]]
            Z, X, Y = data.shape

        buffer = mp.shared_memory.SharedMemory(name=buffer_name)
        shared_memory = np.ndarray(buffer_shape, dtype=buffer_dtype, buffer=buffer.buf)

        for z in range(Z):

            tmp = data[z, :, :] > thrARScl * np.sqrt(varEst)
            tmp = morphology.remove_small_objects(tmp, min_size=minSize, connectivity=4)

            if evtSpatialMask is not None:
                tmp = np.multiply(tmp, evtSpatialMask)

            shared_memory[z, x_range[0]:x_range[1], y_range[0]:y_range[1]] = tmp

        # arLst = measure.label(dActVoxDi)
        # arLst_properties = measure.regionprops(arLst)

        print("DONE: ", x_range, y_range)

    def getAr_h5(path, thrARScl, varEst, minSize, location="dff/neu", evtSpatialMask=None,
                 x_range=None, y_range=None):
        # get data dimensions
        with h5.File(path, "r") as f:
            data = f[location]
            Z, X, Y = data.shape
            cz, cx, cy = data.chunks

        # quality check parameters
        if x_range is None:
            x0 = 0
        else:
            assert x_range[0] % cx == 0, "x range start should be multiple of chunk length {}: {}".format(cx,
                                                                                                          x_range / cx)
            x0, X = x_range

        if y_range is None:
            y0 = 0
        else:
            assert y_range[0] % cy == 0, "y range start should be multiple of chunk length {}: {}".format(cy,
                                                                                                          y_range / cy)
            y0, Y = y_range

        # created shared output array
        binary_map = np.zeros((Z, X, Y), dtype=np.bool_)
        binary_map_shared = shared_memory.SharedMemory(create=True, size=binary_map.nbytes)

        # iterate over chunks
        with mp.Pool(mp.cpu_count()) as pool:

            for chunk_x, chunk_y in list(product(np.arange(x0, X, cx), np.arange(y0, Y, cy))):
                pool.apply(temp,
                           args=(path, [chunk_x, chunk_x + cx], [chunk_y, chunk_y + cy],
                                 thrARScl, varEst, minSize,
                                 binary_map_shared.name, binary_map.dtype, binary_map.shape,
                                 location, evtSpatialMask)
                           )

        # close shared array
        binary_map[:] = np.ndarray(binary_map.shape, dtype=binary_map.dtype, buffer=binary_map_shared.buf)
        binary_map_shared.close()
        binary_map_shared.unlink()
        f.close()

        # get labels
        arLst = measure.label(binary_map)
        # arLst_properties = measure.regionprops(arLst)

        return arLst  # , arLst_properties

    def calculate_noise_h5(data, reference_frame, method="original",
                           chunked=False, ix=None, iy=None, max_x=None, max_y=None):

        # TODO pre-defined variables worth it? --> see after return

        method_options = ["original"]
        assert method in method_options, "method is unknown; options: " + method_options

        if not chunked:
            # > noise estimate original --> similar to standard error!?
            # TODO why exclude the first frame
            dist_squared = np.power(np.subtract(data[1:, :, :], reference_frame), 2)

        else:
            dist_squared = np.power(
                np.subtract(data[1:, 0:max_x, 0:max_y], reference_frame[ix:ix + max_x, iy:iy + max_y]),
                2)  # dim: cz, cx, cy

        # TODO why 0.9133
        dist_squared_median = np.median(dist_squared, 0) / 0.9133  # dim: cx, cy

        return np.power(dist_squared_median, 0.5)


def characterize_event(event_id, t0, t1, data_info, event_info, out_path, split_subevents=True):
    res_path = out_path.joinpath(f"events{event_id}.npy")
    if os.path.isfile(res_path):
        return 2

    # data = tiledb.open(data_path.as_posix())[t0:t1, :, :]
    # event_map = tiledb.open(event_map_path.as_posix())[t0:t1, :, :]

    d_shape, d_dtype, d_name = data_info
    data_buffer = shared_memory.SharedMemory(name=d_name)
    data = np.ndarray(d_shape, d_dtype, buffer=data_buffer.buf)
    data = data[t0:t1, :, :]

    e_shape, e_dtype, e_name = event_info
    event_buffer = shared_memory.SharedMemory(name=e_name)
    event_map = np.ndarray(e_shape, e_dtype, buffer=event_buffer.buf)
    event_map = event_map[t0:t1, :, :]

    if split_subevents:
        z, x, y = np.where(event_map == event_id)
        z0, z1 = np.min(z), np.max(z)
        x0, x1 = np.min(x), np.max(x)
        y0, y1 = np.min(y), np.max(y)

        mask = event_map[z0:z1, x0:x1, y0:y1]
        mask = mask == event_id
        raw = data[z0:z1, x0:x1, y0:y1]

        event_map_sub, _ = detect_subevents(raw, mask)
        event_map = np.zeros(event_map.shape)
        event_map[z0:z1, x0:x1, y0:y1] = event_map_sub

    res = {}
    for em_id in np.unique(event_map):

        if event_id == 0:
            continue

        try:
            event_id_key = f"{event_id}_{em_id}" if split_subevents else event_id
            res[event_id_key] = {}

            z, x, y = np.where(event_map == em_id)
            z0, z1 = np.min(z), np.max(z)
            x0, x1 = np.min(x), np.max(x)
            y0, y1 = np.min(y), np.max(y)

            res[event_id_key]["area"] = len(z)
            res[event_id_key]["bbox"] = ((t0 + z0, t0 + z1), (x0, x1), (y0, y1))

            dz, dx, dy = z1 - z0, x1 - x0, y1 - y0
            z, x, y = z - z0, x - x0, y - y0
            res[event_id_key]["dim"] = (dz + 1, dx + 1, dy + 1)
            res[event_id_key]["pix_num"] = int((dz + 1) * (dx + 1) * (dy + 1))

            mask = np.ones((dz + 1, dx + 1, dy + 1), dtype=np.bool8)
            mask[(z, x, y)] = 0
            res[event_id_key]["mask"] = mask.flatten()
            res[event_id_key]["footprint"] = np.invert(np.min(mask, axis=0)).flatten()

            signal = data[z0:z1 + 1, x0:x1 + 1, y0:y1 + 1]  # TODO weird that i need +1 here
            msignal = np.ma.masked_array(signal, mask)
            res[event_id_key]["trace"] = np.ma.filled(np.nanmean(msignal, axis=(1, 2)))

            res[event_id_key]["error"] = 0

        except ValueError as err:
            print("\t Error in ", event_id_key)
            print("\t", err)
            res[event_id_key]["error"] = 1

    np.save(res_path.as_posix(), res)

    data_buffer.close()
    event_buffer.close()


# DEPRECATED: uses up too much memory
def func_multi(task, data_path, event_path, out_path):
    import numpy as np
    import tiledb

    t0, t1, r0, r1, _ = task
    res_path = out_path.joinpath(f"events{r0}-{r1}.npy")

    if os.path.isfile(res_path):
        return 2

    # load data
    # TODO inefficient
    data = tiledb.open(data_path.as_posix())[t0:t1]  # TODO this gets very big if long events present
    event_map = tiledb.open(event_path.as_posix())[t0:t1]

    print("data: {:.2f} ({}) event: {:.2f} ({})".format(data.nbytes / 1e9, data.dtype, event_map.nbytes / 1e9,
                                                        event_map.dtype))
    return -1

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
            res[event_id]["pix_num"] = int((dz + 1) * (dx + 1) * (dy + 1))

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

    np.save(res_path.as_posix(), res)

    # TODO remove
    # num_files = len(os.listdir(out_path))
    # print("Finished task {}-{} ({:.1f}%)".format(t0, t1, num_files/task_num*100))

    return 1


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
    res["pix_num"] = int((dz + 1) * (dx + 1) * (dy + 1))

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

    return res


def detect_subevents(img, mask, sigma=2,
                     min_local_max_distance=5, local_max_threshold=0.5, min_roi_frame_area=5,
                     verbosity=0):
    assert img.shape == mask.shape, "image ({}) and mask ({}) don't have the same dimension!".format(img.shape,
                                                                                                     mask.shape)

    mask = ~mask
    img = img if sigma is None else gaussian(img, sigma=sigma)  # TODO is this already smoothed?
    masked_img = np.ma.masked_array(img, mask=mask)

    Z, X, Y = mask.shape

    new_mask = np.zeros((Z, X, Y))
    last_mask_frame = np.zeros((X, Y))
    next_event_id = 1
    local_max_container = []
    for cz in range(Z):

        frame_raw = masked_img[cz, :, :]
        frame_mask = mask[cz, :, :]

        if verbosity > 3: print(f"\n----- {cz} -----")

        ####################
        ## Find Local Maxima

        local_maxima = peak_local_max(frame_raw, min_distance=min_local_max_distance, threshold_rel=local_max_threshold)
        local_maxima = np.array(
            [(lmx, lmy, last_mask_frame[lmx, lmy]) for (lmx, lmy) in zip(local_maxima[:, 0], local_maxima[:, 1])],
            dtype="i2")

        # Try to find global maximum if no local maxima were found
        if len(local_maxima) == 0:

            mask_area = np.sum(frame_mask)  # Look for global max
            glob_max = np.unravel_index(np.argmax(frame_raw), (X, Y))  # Look for global max

            if (mask_area > 0) and (glob_max != (0, 0)):
                local_maxima = np.array([[glob_max[0], glob_max[1], 0]])
            else:
                local_max_container.append(local_maxima)
                last_mask_frame = np.zeros((X, Y), dtype="i2")
                continue

        # assign new label to new local maxima (maxima with '0' label)
        for i in range(local_maxima.shape[0]):

            if local_maxima[i, 2] == 0:
                local_maxima[i, 2] = next_event_id
                next_event_id += 1

        # Local Dropouts
        # sometimes local maxima drop below threshold
        # but the event still exists at lower intensity
        # re-introduce those local maxima if the intensity
        # is above threshold value (mask > 0) and the event
        # area of the previous frame is sufficient (area > min_roi_frame_area)
        last_local_max_labels = np.unique(last_mask_frame)
        curr_local_max_labels = np.unique(local_maxima[:, 2])
        if verbosity > 3: print("last_local_max_labels:\n", last_local_max_labels)
        if verbosity > 3: print("curr_local_max_labels:\n", curr_local_max_labels)

        for last_local_max_label in last_local_max_labels:

            if (last_local_max_label != 0) and (last_local_max_label not in curr_local_max_labels):

                prev_local_maxima = local_max_container[-1]
                if verbosity > 5: print("prev_local_maxima: ", prev_local_maxima)
                missing_local_maxima = prev_local_maxima[prev_local_maxima[:, 2] == last_local_max_label]
                prev_area = np.sum(new_mask[cz - 1, :, :] == last_local_max_label)

                # print("missing peak: ", lp, missing_peak)
                if (len(missing_local_maxima) < 1) or (prev_area < min_roi_frame_area):
                    continue

                lmx, lmy, _ = missing_local_maxima[0]
                if ~ frame_mask[lmx, lmy]:  # check that local max still has ongoing event
                    # print("missing peak: ", missing_peak, missing_peak.shape)
                    local_maxima = np.append(local_maxima, missing_local_maxima, axis=0)

        # Local Maximum Uniqueness
        # When a new local maxima appears in a region that was
        # previously occupied, two local maxima receive the same
        # label. Keep label of local maximum closest to previous maximum
        # and assign all local maxima which are further away with
        # new label.
        local_maxima_labels, local_maxima_counts = np.unique(local_maxima[:, 2], return_counts=True)
        for label, count in zip(local_maxima_labels, local_maxima_counts):

            if count > 1:

                # find duplicated local maxima
                duplicate_local_maxima_indices = np.where(local_maxima[:, 2] == label)[0]
                duplicate_local_maxima = local_maxima[local_maxima[:, 2] == label]  # TODO use index instead

                # get reference local maximum
                prev_local_max = local_max_container[-1]
                ref_local_max = prev_local_max[prev_local_max[:, 2] == label]

                # euclidean distance
                distance_to_ref = [np.linalg.norm(local_max_xy - ref_local_max[0, :2]) for local_max_xy in
                                   duplicate_local_maxima[:, :2]]
                min_dist = np.argmin(distance_to_ref)

                # relabel all local maxima that are further away
                to_relabel = list(range(len(distance_to_ref)))
                del to_relabel[min_dist]
                for to_rel in to_relabel:
                    dup_index = duplicate_local_maxima_indices[to_rel]
                    local_maxima[dup_index, 2] = next_event_id
                    next_event_id += 1

        # save current detected peaks
        local_max_container.append(local_maxima)

        #############################
        ## Separate overlaying events

        if local_maxima.shape[0] == 1:

            # Single Local Maximum (== global maximum)
            last_mask_frame = np.zeros((X, Y), dtype="i2")
            last_mask_frame[~frame_mask] = local_maxima[0, 2]

        else:
            # Multiple Local Maxima
            # use watershed algorithm to separate multiple overlaying events
            # with location of local maxima as seeds

            # create seeds
            seeds = np.zeros((X, Y))
            for i in range(local_maxima.shape[0]):
                lmx, lmy, lbl = local_maxima[i, :]
                seeds[lmx, lmy] = lbl

            # run watershed on inverse intensity image
            basin = -1 * frame_raw
            basin[frame_mask] = 0

            last_mask_frame = watershed(basin, seeds).astype("i2")
            last_mask_frame[frame_mask] = 0

        # save results of current run
        new_mask[cz, :, :] = last_mask_frame

    return new_mask, local_max_container


def create_segmentation_video(segmentation, out_path, intensity_image=None, local_maxima=None,
                              cmap_norm=None, non_roi_as_black=True,
                              figsize=(10, 5), dpi=(300), interval=300, repeat_delay=1000):
    num_segments = int(np.max(segmentation))

    if non_roi_as_black and (intensity_image is not None):
        intensity_image[segmentation == 0] = 0

    # create random color map
    if cmap_norm is None:
        r_cmap, r_norm = rand_cmap(num_segments, last_color_black=False)
    else:
        r_cmap, r_norm = cmap_norm

    # create figure
    frames = []
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    ax0.axis('off')
    ax1.axis('off')

    for cz in range(segmentation.shape[0]):

        frame = []
        frame.append(ax1.imshow(segmentation[cz, :, :], animated=True, cmap=r_cmap, norm=r_norm))

        if local_maxima is not None:
            lm = local_maxima[cz]

            if len(lm) > 1:
                frame.append(
                    ax0.scatter(lm[:, 1], lm[:, 0], marker="x", color="red") if len(local_maxima) > 0 else ax0.scatter(
                        0, 0))

        if intensity_image is not None:
            frame.append(ax0.imshow(intensity_image[cz, :, :], animated=True))

        frames.append(frame)

    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True, repeat_delay=repeat_delay)
    ani.save(out_path.as_posix(), dpi=dpi)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap, colors.BoundaryNorm(bounds, nlabels)


if __name__ == "__main__":

    ## arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-t", "--threshold", type=int, default=None)

    args = parser.parse_args()
    args.threshold = args.threshold if args.threshold != -1 else None

    use_dask = True
    subset = None

    if args.input.endswith(".h5"):
        with h5.File(args.input, "r") as file:
            keys = [f"dff/{k}" for k in list(file["dff/"])]

    else:
        keys = ["dff/ast"]

    print("Keys found: {}".format(keys))
    for key in keys:
        print("Starting with : ", key)
        t0 = time.time()

        # run code
        ed = EventDetector(args.input, verbosity=10)
        ed.run(dataset=key, threshold=args.threshold, use_dask=use_dask, subset=subset)

        dt = time.time() - t0
        print("{:.1f} min".format(dt / 60) if dt > 60 else "{:.1f} s".format(dt))
