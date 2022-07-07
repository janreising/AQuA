import os, getopt, sys
import h5py as h5
import dask.array as da
import tiledb
import numpy as np
from dask_image import ndmorph, ndfilters
from skimage import morphology
import scipy.ndimage as ndimage
from tqdm import tqdm

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

def create_event_map(input_path, output_directory, meta_path,
                     use_dask=True, subset=None, adjust_for_noise=False, threshold=0.1, min_size=20):

    event_map_path = f"{output_directory}event_map/"

    if os.path.isdir(event_map_path):
        print("event map already exists. Skipping ... \n\t{}".format(event_map_path))
        return event_map_path

    print("Load data")
    data = _load(input_path, dataset_name=None, use_dask=use_dask, subset=subset)

    print("Estimating noise")
    noise = estimate_background(data) if adjust_for_noise else 1

    print("Thresholding events")
    event_map = get_events(data, roi_threshold=threshold, var_estimate=noise, min_roi_size=min_size)

    print("Saving event map to: ", event_map_path)
    if output_directory is not None:
        event_map.to_tiledb(event_map_path)

    return event_map_path

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

def create_time_map(event_map_path, output_directory):

    time_map_path = f"{output_directory}time_map.npy"
    if os.path.isfile(time_map_path):
        print("time map already exists. Skipping ...")
        return time_map_path

    print("Load event map")
    event_map = da.from_tiledb(event_map_path)

    print("Calculating time map")
    time_map = get_time_map(event_map)

    print("Saving event map to: ", time_map_path)
    if output_directory is not None:
        np.save(time_map_path, time_map)

    return time_map_path

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

    return out

if __name__ == "__main__":

    # GET INPUT
    ifile = None
    loc=None
    output_directory = None
    threshold=0.1
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:o:t:", ["ifile=", "loc=", "output=", "threshold="])
    except getopt.GetoptError:
        print("mapper.py -i <ifile> -l <loc> -o <output> -t <threshold>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            ifile = arg

        if opt in ("-l", "--loc"):
            loc = arg

        if opt in ("-o", "--output"):
            output_directory = arg

        if opt in ("-t", "--threshold"):
            threshold = arg

    # Convert to tileDB if necessary
    if ifile.endswith(".h5"):
        print("h5 file recognized. Converting to .tdb")

        assert loc is not None, ".h5 file recognized. loc parameter must be provided!"

        ifile = export_to_tdb(ifile, loc=loc, out=ifile.replace(".h5", ".tdb"))

    # create event map
    if output_directory is None:
        output_directory = ifile + ".res/"

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
        print("Created dir: ", output_directory)

    event_map_path = create_event_map(ifile, output_directory, None, threshold=threshold)

    # create time map
    time_map_path = create_time_map(event_map_path, output_directory)
