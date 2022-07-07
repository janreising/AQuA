import sys, getopt, os
import numpy as np
import multiprocess as mp
import random
from itertools import repeat
import shutil

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

        # collect events
        dist_events = [np.load(out_path+f, allow_pickle=True)[()] for f in os.listdir(out_path)]

        events = {}
        for e in dist_events:
            events.update(e)
        print("#events: ", len(events.keys()))

        print("Saving results to: ", out_path[:-1]+".npy")
        np.save(out_path[:-1]+".npy", events)

        # removing temp results
        print("Removing temporary results ...")
        shutil.rmtree(out_path)

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
    # print("Working on {}".format(res_path))

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

if __name__ == "__main__":

    idir=None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:", ["idir="])
    except getopt.GetoptError:
        print("mapper.py -i <idir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--idir"):
            idir = arg

    assert idir is not None, "Please provide an input directory! -i --idir"

    print("Calculating features")

    time_map_path = idir + "time_map.npy"
    event_map_path = idir + "event_map/"
    print("Event map: ", event_map_path)
    print("Time map: ", time_map_path)
    custom_slim_features(time_map_path, idir.replace(".res/", ""), event_map_path)

    # print("Saving")
    # with open(f"{idir}/meta.json", 'w') as outfile:
    #     json.dump(meta, outfile)
