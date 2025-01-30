import nd2
import numpy as np
from snakemake.script import snakemake
from tqdm import trange
from tifffile import imwrite
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes

from helpers import END_TIME_CUTOFF


img = nd2.ND2File(snakemake.input.nd2)

times = []
for i in range(img.metadata.contents.frameCount):
    times.append(img.frame_metadata(i).channels[0].time.relativeTimeMs)
exp_times = np.array(times) / 1000 / 60

time_cutoff = np.argwhere(exp_times < END_TIME_CUTOFF).max()

dask = img.to_dask()

gfp = dask[:time_cutoff, 0, ...].compute()

thresholded = np.empty_like(gfp, dtype=bool)
for i in range(gfp.shape[0]):
    threshold = threshold_otsu(gfp[i])
    thresholded[i] = gfp[i] > 0.9 * threshold


def threshold_2d(thresh):
    labels = label(thresh)

    regs = regionprops(labels)
    areas = np.array([reg["area"] for reg in regs])
    ind = np.argmax(areas)

    lbl_id = [reg["label"] for reg in regs][ind]
    lbl = labels == lbl_id
    return binary_fill_holes(lbl)


for i in trange(thresholded.shape[0]):
    thresholded[i] = threshold_2d(thresholded[i])

imwrite(snakemake.output.lbl, thresholded)
img.close()
