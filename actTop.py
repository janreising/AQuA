import argparse
import os
import h5py as h5
import numpy as np
import sys
import scipy.ndimage as ndimage
from scipy.stats import zscore
from skimage import measure


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
    stdEst = calc_noise(data, mskSig)

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

    dARAll = np.zeros((H, W))
    for i, smoI in enumerate(smoVec):

        print(f'Smo {i} ==== \n')
        # opts.smoXY = smoI ## ???

        _, dFSim, sSim = arSimPrep(dSim, smoI)
        _, dFReal, sReal = arSimPrep(data, smoI)

        for j, thrJ in enumerate(thrVec):

            dAR = np.zeros((H, W))
            print(f'Thr {j} \n')

            # null
            tmpSim = dFSim > thrJ*sSim
            szFreqNull = np.zeros((1, H*W))
            for cz in range(dSim.shape[0]):
                tmp00 = np.multiply(tmpSim[cz, :, :], evtSpatialMask)
                cc =


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w  # TODO maybe valid better?


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
            dF[:, ix, iy] = moving_average(data[:, ix, iy], movAvgWin) - xBias

    return dF



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

    import h5py as h
    file = "E:/data/22A5x4/22A5x4-1.zip.h5"

    with h.File(file, "r") as f:
        data = f["cnmfe/ast"][:]

    data = data.astype(np.single)

    arSimPrep(data, 1)
















