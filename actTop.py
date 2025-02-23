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

    def __init__(self, file_path: str,
                 output: str = None, indices: np.array = None, verbosity: int = 1):

        # quality check arguments
        assert os.path.isfile(file_path) or os.path.isdir(file_path), f"input file does not exist: {file_path}"
        assert output is None or ~ os.path.isfile(output), f"output file already exists: {output}"
        assert indices is None or indices.shape == (3, 2), "indices must be np.arry of shape (3, 2) -> ((z0, z1), " \
                                                           "(x0, x1), (y0, y1)). Found: " + indices

        # initialize verbosity
        self.vprint = self.verbosity_print(verbosity)
        self.last_timestamp = time.time()

        # paths
        working_directory = os.sep.join(file_path.split(os.sep)[:-1])
        self.file_path = file_path

        if output is None:
            out_base = ".".join(file_path.split(".")[:-1])
            out_ind = "" if indices is None else "_{}-{}_".format(indices[0], indices[1])

            output_path = f"{out_base}{out_ind}_aqua.h5"
        else:
            output_path = output

        # print settings
        self.vprint(f"working directory: {working_directory}", 1)
        self.vprint(f"input file: {self.file_path}", 1)
        self.vprint(f"output file: {output_path}", 1)

        # Variables
        self.file = None
        self.meta = {}

    def run(self, dataset=None,
            threshold=3, min_size=20, moving_average=25, use_dask=False, adjust_for_noise=False,
            subset=None, output_folder=None):

        self.meta["subset"] = subset
        self.meta["threshold"] = threshold
        self.meta["min_size"] = min_size
        self.meta["adjust_for_noise"] = adjust_for_noise

        # profiling
        pbar = ProgressBar(minimum=5)
        pbar.register()

        # prof = Profiler()
        # prof.register()
        #
        # resources = ResourceProfiler()
        # resources.register()

        # load data
        data = self._load(dataset_name=dataset, use_dask=use_dask, subset=subset)
        # self.data = data
        self.Z, self.X, self.Y = data.shape  # TODO move
        self.vprint(data if use_dask else data.shape, 2)

        noise = self.estimate_background(data) if adjust_for_noise else 1

        # event_map, event_properties = self.get_events(data, roi_threshold=threshold, var_estimate=noise,
        #                                               min_roi_size=min_size)

        event_map = self.get_events(data, roi_threshold=threshold, var_estimate=noise, min_roi_size=min_size)

        if output_folder is not None:
            # np.save(f"{output_folder}/event_map.npy", event_map)
            da.to_npy_stack(f"{output_folder}/event_map/", event_map, axis=0)

        # event_map = dask.compute(event_map)  # TODO probably not the right place to do this

        # getting rid of extra variables
        self.vprint("collecting garbage", 2)
        del data
        del noise
        gc.collect()
        time.sleep(10)

        self.vprint("calculating features", 2)
        events = self.custom_slim_features(event_map)

        # return event_map, data

        # return event_map, event_properties
        # self.event_map, self.event_properties = event_map, event_properties

        # meta, meta_lookup_tbl, raw_trace_store, mask_store, _ = self.dummy_slim_features(event_properties, output_folder)
        # self.meta, self.meta_lookup_tbl, self.raw_trace_store, self.mask_store, self.footprints = meta, meta_lookup_tbl, raw_trace_store, mask_store, footprints
        # self.vprint("features extracted", 3)

        # return None

        if output_folder is not None:

            np.save(f"{output_folder}/events.npy", events)

            with open(f"{output_folder}/meta.json", 'w') as outfile:
                json.dump(self.meta, outfile)

            self.vprint("features saved", 3)

        # # stop profiling
        # # for r in prof.results:
        # #     self.vprint(r, 2)
        # prof.clear()
        # prof.unregister()
        #
        # # for r in resources.results:
        # #     self.vprint(r, 2)
        # resources.clear()
        # resources.unregister()

        self.vprint("Run complete!", 1)

        # return event_map, raw_trace_store, mask_store, footprints, meta

        # features = self.calculate_event_features(data, event_map, event_properties, 1, moving_average, threshold)
        # features = self.calculate_event_propagation(data, event_properties, features)

    def verbosity_print(self, verbosity_level: int):

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

        if self.file_path.endswith(".h5"):

            assert dataset_name is not None, "'dataset_name' required if providing an hdf file"

            file = h5.File(self.file_path, "r")
            assert dataset_name in file, "dataset '{}' does not exist in file".format(dataset_name)

            data = da.from_array(file[dataset_name], chunks='auto') if use_dask else file[dataset_name]

        elif self.file_path.endswith(".tdb"):

            data = tiledb.open(self.file_path)

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

    def get_events(self, data: np.array, roi_threshold: float, var_estimate: float,
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
            event_map = np.zeros(data.shape, dtype=np.uint16)
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


        del active_pixels   # TODO does this actually help?

        # characterize each event

        # event_properties = measure.regionprops(event_map, intensity_image=data, cache=True,
        #                                        extra_properties=[self.trace, self.footprint]
        #                                        )
        # self.vprint("events collected", 3)

        return event_map#, event_properties

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
        bboxes = np.array([[p.bbox[0], p.bbox[3], p.bbox[1], p.bbox[4], p.bbox[2], p.bbox[5]] for p in event_properties])

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

        return meta, meta_lookup_tbl, raw_trace_store, mask_store, None#, footprints

    def save_slim_features(self, event_properties, output_folder):

        num_events = len(event_properties)
        bboxes = np.array([[p.bbox[0], p.bbox[3], p.bbox[1], p.bbox[4], p.bbox[2], p.bbox[5]] for p in event_properties])

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
        summary_keys = ["raw_trace_ind_0", "raw_trace_ind_1", "mask_ind_0", "mask_ind_1", "label", "area", "bbox_z0", "bbox_z1", "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1"]
        self.meta["summary_columns"] = summary_keys

        memmap_summary = np.memmap(f"{output_folder}/summary.mmap", mode="w+", shape=(num_events, len(summary_keys)), dtype=np.int32)
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

    def custom_slim_features(self, event_map, parallel=True):

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
            num_task = num_events-1

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
    print("nBytes: {} ... {}GB ... max: {}-{}".format(A.nbytes, A.nbytes/1e9, np.iinfo(dtype).min, np.iinfo(dtype).max))

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


def func(event_id, shape, sh_event_name, file_path):
    from multiprocess import shared_memory
    import numpy as np
    import tiledb

    res = {}
    res["label"] = event_id

    em = shared_memory.SharedMemory(name=sh_event_name)
    em_np = np.ndarray(shape, dtype='uint16', buffer=em.buf)

    # d = shared_memory.SharedMemory(name=sh_data_name)
    # d_np = np.ndarray(shape, dtype='float32', buffer=d.buf)

    data = tiledb.open(file_path)

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

    return res

if __name__ == "__main__":
    """
    ## arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--indices", "--ind", type=int, nargs=2, default=None)
    parser.add_argument("--inMemory", "--mem", action='store_true')
    parser.add_argument("-v", "--verbose", type=int, default=0)

    parser.add_argument("--rl", "--raw_location", type=str, default="cnmfe/ast")
    parser.add_argument("--dl", "--df_location", type=str, default="dff/ast")

    parser.add_argument("--frameRate", type=float, default=1.0, help="frame rate images/s")
    parser.add_argument("--spatialRes", type=float, default=0.5, help="spatial resolution µm/px")
    # parser.add_argument("--varEst", type=float, default=0.02, help="Estimated noise variance")

    parser.add_argument("--minSize", type=int, default=10, help="minimum ROI size")
    parser.add_argument("--smoXY", type=int, default=1, help="spatial smoothing level")
    parser.add_argument("--thrARScl", type=int, default=1, help="active voxel threshold")

    args = parser.parse_args()

    # Create AqUA object
    c = ActTop(args)
    # c.run()
    """

    t0 = time.time()

    use_small = False
    use_dask = True
    use_subset = False

    subset = None if use_small or not use_subset else [0, 100, None, None, None, None]
    print("subset: ", subset)

    # file path
    directory = "C:/Users/janrei/Desktop/"
    file = "22A5x4-1.zip.h5" if use_small else "22A5x4-2.subtr.reconstr.mc.tdb"  # "22A5x4-2.zip.h5.tdb"
    loc = "/dff/neu" if use_small else "/dff/ast/"
    path = directory + file

    # output = 'C:/Users/janrei/Desktop/22A5x4-2.zip.h5.tdb'

    ed = EventDetector(path, verbosity=10)
    ed.run(dataset=loc, threshold=0.1, use_dask=use_dask, subset=subset,
                 output_folder="C:/Users/janrei/Desktop/22A5x4-2.subtr.reconstr.res/"
                 )

    dt = time.time() - t0
    print("{:.1f} min".format(dt / 60) if dt > 60 else "{:.1f} s".format(dt))

    sys.exit(2)

    # # event_map = event_map > 1
    # spacer = np.ones((600, 1))
    # subset = [0, event_map.shape[0]] if subset is None else subset
    # plt.imshow(np.concatenate((event_map[0, :, :], spacer,
    #                            event_map[int(subset[1] / 2), :, :], spacer,
    #                            event_map[subset[1] - 5, :, :]),
    #                           axis=1))
    # plt.show()

    print("Done!")

    ep = ed.event_properties
    ep[0].area


    # print("Saving")
    # evt_dask = da.from_array(event_map)
    # evt_dask.to_tiledb(path+"evt_map.tdb")

    smoXY = 1
    thr = 1

    ed = EventDetector(path, verbosity=10)
    data = ed._load(dataset_name=loc, use_dask=True, subset=subset)
    p(data, data)
    noise = ed.estimate_background(data).compute()
    print("Noise: {} [{}]".format(noise, thr * noise))

    active_pixels = ndfilters.gaussian_filter(data, smoXY) > thr  # * noise
    ap2 = active_pixels.copy()
    struct = ndimage.generate_binary_structure(3, 4)
    print(struct.shape, "\n", struct)
    ap2 = ndmorph.binary_opening(ap2, structure=struct)
    ap2 = ndmorph.binary_closing(ap2, structure=struct)
    p(active_pixels, ap2)

    tf.imsave(path + ".del.tiff", data)
    tf.imsave(path + ".del1.tiff", active_pixels)
    tf.imsave(path + ".del2.tiff", ap2)
    data_masked = data.copy()
    data_masked[ap2 > 0] = 0
    tf.imsave(path + ".del3.tiff", data_masked)


    # ragged array
    num_events = len(event_properties)
    bboxes = np.array([[p.bbox[0], p.bbox[3], p.bbox[1], p.bbox[4], p.bbox[2], p.bbox[5]] for p in event_properties])


    elength = bboxes[:, 1]-bboxes[:, 0]
    eindices = np.cumsum(elength)  # indices for each event trace
    raw_traces = da.from_array(np.zeros(np.sum(elength), dtype=np.single))

    for i, p in enumerate(event_properties):

        i0 = eindices[i-1] if i > 0 else 0
        i1 = eindices[i]

        masked = da.ma.masked_array(p.intensity_image, p.intensity_image == 0)
        raw_traces[i0:i1] = da.ma.filled(np.nanmean(masked, axis=(1, 2)))

    raw_traces = raw_traces.compute()
    print(type(raw_traces))

    # multiprocessing
    def func(event):
        area = event["area"]
        print(area)

    with mp.Pool(mp.cpu_count()) as p:
        p.map(func, ep[:24])
