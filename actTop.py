import argparse
import os
import h5py as h5
import numpy as np
import sys
import time
import scipy.ndimage as ndimage
from scipy.stats import zscore
from skimage import measure
from skimage import morphology
import multiprocessing as mp
from itertools import product
from multiprocessing import shared_memory
from pathos.multiprocessing import ProcessingPool as Pool

class ActTop():

    def __init__(self, args):

        ## quality check arguments
        assert os.path.isfile(args.input), f"input file does not exist: {args.input}"
        assert args.output is None or ~ os.path.isfile(args.output), f"output file already exists: {args.output}"

        ## initialize verbosity
        global vprint  # TODO bad programming !?
        vprint = self.verbosity_print(args.verbose)

        ## paths
        working_directory = os.sep.join(args.input.split(os.sep)[:-1])
        self.file_path = args.input
        feature_path = ".".join(args.input.split(".")[:-1]) + "_FeatureTable.xlsx"

        if args.output is None:
            out_base = ".".join(args.input.split(".")[:-1])
            out_ind = "" if args.indices is None else "_{}-{}_".format(args.indices[0], args.indices[1])

            output_path = f"{out_base}{out_ind}_aqua.h5"
        else:
            output_path = args.output

        ## print settings
        vprint(f"working directory: {working_directory}", 1)
        vprint(f"input file: {self.file_path}", 1)
        vprint(f"output file: {output_path}", 1)
        vprint(f"feature file: {feature_path}", 1)

        v = vars(parser.parse_args())
        for key in v.keys():
            vprint("\t{}:\t{}".format(key, v[key]), 2)

        ## Variables
        self.file = None

    def run(self):

        raw, dff = self.load_data(self.file_path, args.raw_location, args.df_location, in_memory=args.inMemory)
        self.func_actTop(raw, dff, in_memory=args.inMemory)

        if self.file is not None:
            self.file.close()

    @staticmethod
    def verbosity_print(verbosity_level):

        # TODO add timings

        def vlevel_print(msg, verbosity):
            if verbosity <= verbosity_level:
                print("\t"*(verbosity-1) + "*"*(verbosity) + " " + msg)

        return vlevel_print

    def load_data(self, file_path, raw_location, df_location, in_memory=True):

        # TODO one should be on cnmfe / one should be on dF

        file = h5.File(file_path, "r")
        self.file = file

        for loc in [raw_location, df_location]:
            assert loc in file, f"unable to find dataset {loc} in {file_path}"

        if in_memory:
            raw = file[raw_location][:]
            dff = file[df_location][:]
        else:
            raw = file[raw_location]
            dff = file[df_location]

        if in_memory:
            file.close()

        return raw, dff

    def func_actTop(self, raw, dff, foreground_threshold=0, in_memory=True,
                        noise_estimation_method = "original"):

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

                    max_x = min(X-ix, cx)  # at most load till X border of image
                    max_y = min(Y-iy, cy)  # at most load till Y border of image

                    chunk[:, 0:max_x, 0:max_y] = raw[:, ix:ix+max_x, iy:iy+max_y]  # load chunk

                    mean_project[ix:ix+max_x, iy:iy+max_y] = np.mean(chunk, 0)
                    var_project[ix:ix+max_x, iy:iy+max_y] = np.var(chunk, 0)

                    noise_map[ix:ix+max_x, iy:iy+max_y] = self.calculate_noise(
                        chunk, reference_frame, chunked=True,
                        ix=ix, iy=iy, max_x=max_x, max_y=max_y
                    )

        # > Create masks
        msk000 = var_project > 1e-8  # non-zero variance # TODO better variable name

        msk_thrSignal = mean_project > foreground_threshold
        noiseEstMask = np.multiply(msk000, msk_thrSignal)   # TODO this might be absolutely useless for us

        # > noise level
        vprint("calculating noise ...", 1)
        sigma = 0.5
        noise_map_gauss_before = ndimage.gaussian_filter(noise_map, sigma=sigma,
                                                         truncate=np.ceil(2*sigma)/sigma, mode="wrap")
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


# def calculate_noise(data, reference_frame, method="original",
#                         chunked=False, ix=None, iy=None, max_x=None, max_y=None):
#
#     # TODO pre-defined variables worth it? --> see after return
#
#     method_options = ["original"]
#     assert method in method_options, "method is unknown; options: " + method_options
#
#     if not chunked:
#         #> noise estimate original --> similar to standard error!?
#         #TODO why exclude the first frame
#         dist_squared = np.power(np.subtract(data[1:, :, :], reference_frame), 2)
#
#     else:
#         dist_squared = np.power(
#                 np.subtract(data[1:, 0:max_x, 0:max_y], reference_frame[ix:ix+max_x, iy:iy+max_y]),
#             2)  # dim: cz, cx, cy
#
#     # TODO why 0.9133
#     dist_squared_median = np.median(dist_squared, 0) / 0.9133  # dim: cx, cy
#
#     return np.power(dist_squared_median, 0.5)

def calc_noise(data, mskSig=None):

    xx = np.power(data[1:, :, :] - data[:-1, :, :], 2)  # dim: Z, X, Y
    stdMap = np.sqrt(np.median(xx, 0) / 0.9133)  # X, Y

    if mskSig is not None:
        stdMap[~mskSig] = None

    stdEst = np.nanmedian(stdMap)  # dim:1

    return stdEst


def arSimPrep(data, smoXY):

    mskSig = np.var(data, 0) > 1e-8

    dat = data.copy()  # TODO inplace possible?
    dat = dat + np.random.randn(*dat.shape)*1e-6

    # smooth data
    if smoXY > 0:
        for z in range(dat.shape[0]):
            dat[z, :, :] = ndimage.gaussian_filter(dat[z, :, :], sigma=smoXY, truncate=np.ceil(2*smoXY)/smoXY, mode="wrap")

    # estimate noise
    stdEst = calc_noise(dat, mskSig)

    dF = getDfBlk(dat, cut=200, movAvgWin=25, stdEst=stdEst)  # TODO cut, movAvgWin variable

    return dat, dF, stdEst

def getARSim(data, smoMax, thrMin, minSize, evtSpatialMask=None,
             smoCorr_location="smoCorr.h5"):

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
                                        truncate=np.ceil(2*sigma)/sigma, mode="wrap")

    rto = data.shape[0] / dSim.shape[0]

    # > simulation
    smoVec = [smoMax]  # TODO looks like this should be a vector!?!?!?!
    thrVec = np.arange(thrMin+3, thrMin-1, -1)

    t0 = time.time()
    dARAll = np.zeros((H, W))
    for i, smoI in enumerate(smoVec):

        print("Smo {} [{}] ==== - {:.2f}\n".format(i, smoI, (time.time()-t0)/60))
        # opts.smoXY = smoI ## ???

        _, dFSim, sSim = arSimPrep(dSim, smoI)
        _, dFReal, sReal = arSimPrep(data, smoI)

        for j, thrJ in enumerate(thrVec):

            dAR = np.zeros((T, H, W))
            print("Thr{}:{} [> {:.2f}| >{:.2f}]  - {:.2f}\n".format(j, thrJ, thrJ*sSim, thrJ*sReal, (time.time()-t0)/60))

            # > null
            print("\t calculating null ... - {:.2f}".format((time.time()-t0)/60))
            tmpSim = dFSim > thrJ*sSim
            szFreqNull = np.zeros((H*W))
            for cz in range(T):     # TODO shouldn't be range T but 200!?
                # tmp00 = np.multiply(tmpReal[cz, :, :], evtSpatialMask)
                tmp00 = tmpSim[cz, :, :]

                # identify distinct areas
                labels = measure.label(tmp00)  # dim X.Y, int
                areas = [prop.area for prop in measure.regionprops(labels)]

                for area in areas:
                    szFreqNull[area] += 1

            szFreqNull = szFreqNull*rto

            # > observation
            print("\t calculating observation ... - {:.2f}".format((time.time()-t0)/60))
            tmpReal = dFReal > thrJ*sReal
            szFreqObs = np.zeros(H*W)  # counts frequency of object size
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
            print("\t applying to data ... - {:.2f}".format((time.time()-t0)/60))
            if suc > 0:

                e00 = int(np.ceil(smoI/2))
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


def getAr(data, thrARScl, varEst, minSize, evtSpatialMask=None):

    Z, X, Y = data.shape

    dActVoxDi = np.zeros(data.shape, dtype=np.bool8)
    for z in range(Z):

        tmp = data[z, :, :] > thrARScl * np.sqrt(varEst)
        tmp = morphology.remove_small_objects(tmp, min_size=minSize, connectivity=4)

        if evtSpatialMask is not None:
            tmp = np.multiply(tmp, evtSpatialMask)

        dActVoxDi[z, :, :] = tmp

    arLst = measure.label(dActVoxDi)
    arLst_properties = measure.regionprops(arLst, intensity_image=data)  # TODO split if necessary

    return arLst, arLst_properties

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
        assert x_range[0] % cx == 0, "x range start should be multiple of chunk length {}: {}".format(cx, x_range / cx)
        x0, X = x_range

    if y_range is None:
        y0 = 0
    else:
        assert y_range[0] % cy == 0, "y range start should be multiple of chunk length {}: {}".format(cy, y_range / cy)
        y0, Y = y_range

    # created shared output array
    binary_map = np.zeros((Z, X, Y), dtype=np.bool_)
    binary_map_shared = shared_memory.SharedMemory(create=True, size=binary_map.nbytes)

    # iterate over chunks
    with mp.Pool(mp.cpu_count()) as pool:

        for chunk_x, chunk_y in list(product(np.arange(x0, X, cx), np.arange(y0, Y, cy))):
            pool.apply(temp,
                       args=(path, [chunk_x, chunk_x+cx], [chunk_y, chunk_y+cy],
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

    return arLst#, arLst_properties

def moving_average(a, w):
    y = ndimage.filters.uniform_filter1d(a, size=w)
    return y


def moving_average_xy(a, w):

    _, X, Y = a.shape

    res = np.zeros(a.shape)
    for x, y in list(zip(range(X), range(Y))):

        res[:, x, y] = ndimage.filters.uniform_filter1d(a[:, x, y], size=w)

    return res


def getLmAll(data, arLst, arLst_properties, fsz=(1, 1, 0.5), bounding_box_extension=3):

    """ Detect all local maxima in the video

    :param data:
    :param arLst:
    :param arLst_properties:
    :param fsz:
    :param bounding_box_extension:
    :return:
    """

    Z, X, Y = data.shape

    # detect in active regions only
    # lmAll = np.zeros(data.shape, dtype=np.bool_)
    lmLoc = []  # location of local maxima
    lmVal = []  # values of local maxima
    for i, prop in enumerate(arLst_properties):

        # pix0 = prop.coords  # absolute coordinates
        bbox_z0, bbox_x0, bbox_y0, bbox_z1, bbox_x1, bbox_y1 = prop.bbox  # bounding box coordinates
        # pix0_relative = pix0 - (bbox_z0, bbox_x0, bbox_y0)  # relative coordinates

        # extend bounding box
        bbox_z0 = max(bbox_z0-bounding_box_extension, 0)
        bbox_z1 = min(bbox_z1+bounding_box_extension, Z)

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
            mskST = arLst[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1] > 0
            dInst = data[bbox_z0:bbox_z1, bbox_x0:bbox_x1, bbox_y0:bbox_y1]
            mskSTSeed = np.multiply(dInst, mskST)

        # get local maxima
        dInst_smooth = ndimage.gaussian_filter(dInst, sigma=fsz)
        local_maxima = morphology.local_maxima(dInst_smooth, allow_borders=False, indices=True)
        lm_z, lm_x, lm_y = local_maxima

        # select local maxima within mask
        for lmax in zip(lm_z, lm_x, lm_y):

            if mskST[lmax] > 0:
                lmLoc.append(lmax)
                lmVal.append(mskSTSeed[lmax])

        return lmLoc, lmVal


def getFeaturesTop(data, evtMap, arLst_properties,
                   frame_rate, spatial_resolution, movAvgWin,
                   fit_option="exp1", fit_max_iter=100,
                   use_poissoin_noise_model=True,
                   skip_single_frame_events=True):

    ## arLst eventMap

    Z, X, Y = data.shape
    if use_poissoin_noise_model:
        data = np.sqrt(data)

    # datx = data.copy()  # TODO really necessary?
    # datx[evtMap > 0] = None  # set all detected events None
    # datx = imputeMov(datx)

    Tww = min(movAvgWin, Z/4)
    bbm = 0

    ftsLst = {}
    ftsLst["baisc"] = []
    ftsLst["propagation"] = []

    dMat = np.zeros((len(arLst_properties), Z, 2), dtype=np.single)
    dffMat = np.zeros((len(arLst_properties), Z, 2), dtype=np.single)

    for i, evt in enumerate(arLst_properties):

        pix0 = evt.coords  # absolute coordinates
        bbox_z0, bbox_x0, bbox_y0, bbox_z1, bbox_x1, bbox_y1 = evt.bbox  # bounding box coordinates
        pix0_relative = pix0 - (bbox_z0, bbox_x0, bbox_y0)  # relative coordinates

        mskST = evt.image  # binary mask of active region
        mskSTSeed = evt.intensity_image  # real values of region (equivalent to dInst .* mskST)

        # skip if event is only one frame long
        if skip_single_frame_events and (bbox_z0 == bbox_z1):
            continue

        xy_footprint = np.max(mskST, axis=0)
        dInst = data[:, bbox_x0:bbox_x1, bbox_y0:bbox_y1]  # bounding box intensity

        raw_curve = np.mean(dInst, axis=(1, 2), where=xy_footprint)

        # TODO event_indices have to include all events detected

    return ftsLst, dffMat, dMat

def curvePolyDeTrend(curve, event_indices):

    X = range(len(curve))
    Y = curve






def moving_average_h5(path, loc, output, x, y, w,
                      shmem_shape, shmem_dtype):

    with h5.File(path, "r") as file:

        data = file[loc]
        cz, cx, cy = data.chunks

        chunk = data[:, x:x+cx, y:y+cy]
        Z, X, Y = chunk.shape

    buffer = mp.shared_memory.SharedMemory(name=output)
    dF = np.ndarray(shmem_shape, dtype=shmem_dtype, buffer=buffer.buf)

    for xi in range(X):
        for yi in range(Y):
            dF[:, x+xi, y+yi] = ndimage.filters.uniform_filter1d(chunk[:, xi, yi], size=w)


def getDfBlk(data, cut, movAvgWin, stdEst=None, spatialMask=None):

    # cut: frames per segment

    if stdEst is None:
        stdEst = calc_noise(data, spatialMask)

    T, H, W = data.shape

    dF = np.zeros(data.shape, dtype=np.single)

    # > calculate bias
    num_trials = 10000
    xx = np.random.randn(num_trials, cut)*stdEst

    xxMA = np.zeros(xx.shape)
    for n in range(num_trials):
        xxMA[n, :] = moving_average(xx[n, :], movAvgWin)

    xxMin = np.min(xxMA, axis=1)
    xBias = np.nanmean(xxMin)

    # > calculate dF
    for ix in range(H):
        for iy in range(W):
            dF[:, ix, iy] = data[:, ix, iy] - min(moving_average(data[:, ix, iy], movAvgWin)) - xBias

    return dF


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
    xx = np.random.randn(num_trials, cut)*stdEst

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


import matplotlib.pyplot as plt
def show(A, label=None):

    if type(A) == list:

        fig, axx = plt.subplots(1, len(A))

        for ax, a in list(zip(axx, A)):
            ax.imshow(a)
            if label is not None: ax.set_title(label)

    else:
        plt.imshow(A)

    plt.show()


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

    import h5py as h5
    import numpy as np
    file = "E:/data/22A5x4/22A5x4-1.zip.h5"
    # file = "C:/Users/janrei/Desktop/22A5x4-2.zip.h5"

    cut = 200
    movWin = 20

    run_original = True

    """
    ## SHORT FILE
    print("-- short file --")
    with h.File(file, "r") as f:
        data = f["cnmfe/neu"][0:200, 0:200, 0:200]
        # data = data.astype(np.single)
        print("shape: ", data.shape)

    t0 = time.time()
    r0 = getDfBlk(data, cut, movWin)
    t1 = time.time()

    print("Original ...\t{:.2f}s".format(t1 - t0))

    r1 = getDfBlk_faster(data, cut, movWin)
    t2 = time.time()
    print("Optimized ...\t{:.2f}s".format(t2 - t1))

    print("\nDIFF: \t{:.3f}s [{:.1f}%]".format((t1 - t0) - (t2 - t1), (t2 - t1) / (t1 - t0) * 100))

    same = np.allclose(r0, r1, atol=0.1)
    print("Results equivalent: {}".format(same))
    

    ## LONG FILE
    print("\n\n-- long file --")

    X_max = 300
    Y_max = 300

    t0 = time.time()

    if run_original:
        with h.File(file, "r") as f:
            data = f["cnmfe/neu"][:, 0:X_max, 0:Y_max]
            # data = data.astype(np.single)
            print("shape: ", data.shape)
        r0 = getDfBlk(data, cut, movWin, stdEst=0.1)
    else:
        time.sleep(1)
        r0 = None

    t1 = time.time()

    print("Original ...\t{:.2f}s".format(t1-t0))

    r1 = getDfBlk_faster(file, "cnmfe/neu", cut, movWin, x_max=X_max, y_max=Y_max, stdEst=0.1)
    t2 = time.time()
    print("Optimized ...\t{:.2f}s".format(t2-t1))

    print("\nDIFF: \t{:.3f}s [{:.1f}%]".format((t1-t0) - (t2-t1), (t2-t1)/(t1-t0)*100))

    same = np.allclose(r0, r1, atol=0.1)
    print("Results equivalent: {}".format(same))
    
    """

    # with h5.File(file, "r") as f:
    #     data = f["dff/neu"][:cut, :, :]

    # noise = calc_noise(data)
    # print("Noise: ", noise)

    with h5.File(file, "r") as f:
        data = f["dff/neu"][:]

    stdEst = calc_noise(data)

    arLst, arLst_properties = getAr(data, 3, stdEst, 10, evtSpatialMask=None)
    print(len(arLst_properties))
    show(arLst[50, :, :])

    # print(len(arLst))

    i = 64
    evt = arLst_properties[i]

    n = 5
    show([mskST[n, :, :], dInst[n, :, :], mskSTSeed[n, :, :], dInst_smooth[n, :, :]])

    z_last = -1
    for z, x, y in lmLoc:
        tmp_ = mskSTSeed[z, :, :].copy() if z_last != z else tmp_
        tmp_[x, y] = 2

        show([mskSTSeed[z, :, :], tmp_], label=z)
        # show(mskSTSeed[z, :, :])

        z_last = z























