import nd2
import pandas as pd
from tifffile import imread, imwrite

from edt import edt
import numpy as np
from snakemake.script import snakemake
from tqdm import trange

from helpers import time_col, gfp_col, dist_um_col, END_TIME_CUTOFF


lbl = imread(snakemake.input.lbl).astype(bool)

img = nd2.ND2File(snakemake.input.nd2)
dask = img.to_dask()

times = []
for i in range(img.metadata.contents.frameCount):
    times.append(img.frame_metadata(i).channels[0].time.relativeTimeMs)
exp_times = np.array(times) / 1000 / 60

time_cutoff = np.argwhere(exp_times < END_TIME_CUTOFF).max()
if "HMBR" in snakemake.wildcards.file:
    time_cutoff = np.argwhere(exp_times < 60).max()
    lbl = np.concatenate(
        [lbl, np.repeat(lbl[-1:], time_cutoff - lbl.shape[0], axis=0)],
        axis=0,
    )
exp_times = exp_times[:time_cutoff]

pixel_size = img.metadata.channels[0].volume.axesCalibration[0]

distances = np.empty_like(lbl, dtype=float)
dfs = []

for i in trange(distances.shape[0]):
    gfp = dask[i, 0, ...].compute()

    tmp = lbl[i]
    distances[i] = edt(tmp)
    dists = distances[i][tmp]

    dfs.append(
        pd.DataFrame(
            {
                gfp_col: gfp[tmp],
                dist_um_col: dists * pixel_size,
                time_col: exp_times[i],
            }
        )
    )
df = pd.concat(dfs, ignore_index=True)

print("before writing output")
if "HMBR" in snakemake.wildcards.file:
    imwrite(snakemake.output.edt, distances[:1].astype(np.float32) * pixel_size)
else:
    imwrite(snakemake.output.edt, distances.astype(np.float32) * pixel_size)

print("before deletion")
img.close()
del gfp
del distances
print("after deletion")

NUM_BINS = 100
RUNNING_MEAN_N = 5


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def running_mean_error(x, N):
    cumsum = np.cumsum(np.insert(x**2, 0, 0))
    return np.sqrt(cumsum[N:] - cumsum[:-N]) / float(N)


def derivative(x):
    return x[1:-1], (x[2:] - x[:-2]) / 2


def derivative_time(x, t):
    t_diff = t[2:] - t[:-2]
    return x[1:-1], (x[2:] - x[:-2]) / t_diff


def derivative_error(x):
    return x[1:-1], np.sqrt(x[2:] ** 2 + x[:-2] ** 2) / 2


def derivative_error_time(x, t):
    t_diff = t[2:] - t[:-2]
    return x[1:-1], np.sqrt(x[2:] ** 2 + x[:-2] ** 2) / t_diff


exp_times = np.sort(df[time_col].unique())

df["bins"] = pd.cut(df[dist_um_col], bins=NUM_BINS)

binned_df_ = df.groupby([time_col, "bins"])[gfp_col].agg(["mean", "std"])
binned_df = binned_df_.reset_index()
binned_df[dist_um_col] = binned_df["bins"].apply(lambda x: x.mid).astype(float)
x = np.sort(binned_df[dist_um_col].unique())

print("after pandas binning")

heatmap_mean = binned_df_["mean"].unstack().to_numpy()
heatmap_std = binned_df_["std"].unstack().to_numpy()

t_deriv_mean = np.empty(
    (heatmap_mean.shape[0] - RUNNING_MEAN_N - 1, heatmap_mean.shape[1])
)
t_deriv_std = np.empty_like(t_deriv_mean)

running_mean_time = running_mean(exp_times, RUNNING_MEAN_N)

for i in range(t_deriv_mean.shape[1]):
    _, t_deriv_mean[:, i] = derivative_time(
        running_mean(heatmap_mean[:, i], RUNNING_MEAN_N), running_mean_time
    )
    _, t_deriv_std[:, i] = derivative_error_time(
        running_mean_error(heatmap_std[:, i], RUNNING_MEAN_N), running_mean_time
    )

deriv_time = running_mean_time[1:-1]

np.save(snakemake.output.i_mean, heatmap_mean)
np.save(snakemake.output.i_std, heatmap_std)
np.save(snakemake.output.deriv_mean, t_deriv_mean)
np.save(snakemake.output.deriv_std, t_deriv_std)

np.save(snakemake.output.exp_times, exp_times)
np.save(snakemake.output.deriv_time, deriv_time)
np.save(snakemake.output.x, x)
