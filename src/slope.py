import yaml
from snakemake.script import snakemake
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

heatmap = np.load(snakemake.input.heatmap)
time = np.load(snakemake.input.time)
distance = np.load(snakemake.input.x)

end_time_cutoff = 6
if snakemake.wildcards.file == "TMP-FAST_50um_2":
    end_time_cutoff = 4
if "HMBR" in snakemake.wildcards.file:
    end_time_cutoff = 60
time_cutoff = np.argwhere(time < end_time_cutoff).max()

time = time[:time_cutoff]
heatmap = heatmap[:time_cutoff, :]

t_max = np.nanmax(heatmap, axis=1)

inds = np.logical_not(np.any(np.isnan(heatmap), axis=0))
argmax_dist = np.argmax(heatmap[:, inds], axis=0)
max_time = time[argmax_dist]

x_values = distance[inds]
max_dist = x_values.max()

if "HMBR" in snakemake.wildcards.file:
    inds = x_values < 30
    x_values = x_values[inds]
    max_time = max_time[inds]

res = linregress(x_values, max_time)
# res = linregress(max_time, x_values)

with open(snakemake.output.yaml, "w") as f:
    yaml.dump(
        {
            "slope": float(res.slope),
            "slope_err": float(res.stderr),
            "max_dist": float(max_dist),
        },
        f,
    )

X, Y = np.meshgrid(distance, time)
plt.pcolormesh(X, Y, heatmap, shading="nearest")
plt.scatter(x_values, max_time, marker="x", alpha=0.5, color="black")

# X, Y = np.meshgrid(time, distance)
# plt.pcolormesh(X, Y, heatmap.T, shading="nearest")
# plt.scatter(max_time, x_values, marker="x", alpha=0.5, color="black")

x0, x1 = plt.xlim()
y0_, y1_ = plt.ylim()

y0 = x0 * res.slope + res.intercept
y1 = x1 * res.slope + res.intercept

plt.plot((x0, x1), (y0, y1), alpha=0.5, color="black")

plt.xlim((x0, x1))
plt.ylim((y0_, y1_))

plt.savefig(snakemake.output.png)
