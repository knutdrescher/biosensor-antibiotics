import yaml
from snakemake.script import snakemake
import numpy as np
import matplotlib.pyplot as plt


heatmap = np.load(snakemake.input.heatmap)
time = np.load(snakemake.input.time)
distance = np.load(snakemake.input.x)

end_time_cutoff = 10
if "HMBR" in snakemake.wildcards.file:
    end_time_cutoff = 30
time_cutoff = np.argwhere(time < end_time_cutoff).max()

time = time[:time_cutoff]
heatmap = heatmap[:time_cutoff, :]

finite = np.all(np.isfinite(heatmap), axis=0)
assert finite.shape == distance.shape

finite_distance = distance[finite]
perc = np.percentile(finite_distance, q=[5, 95])


def perc_norm(c, low=2.5, high=97.5):
    perc = np.percentile(c, q=[low, high])
    div = perc[1] - perc[0]
    return (c - perc[0]) / div, div


low_inds = distance < perc[0]
high_inds = distance > perc[1]

curves = []
stds = []
for inds in [low_inds, high_inds]:
    tmp = heatmap[:, inds]
    curve, div = perc_norm(np.nanmean(tmp, axis=1))
    std = np.nanstd(tmp, axis=1) / div
    curves.append(curve)
    stds.append(std)

times = [time[curve > 0.5].min() for curve in curves]
times.append(time[curves[0] > 0.05].min())

low_times = [time[curve - std > 0.5].min() for curve, std in zip(curves, stds)]
low_times.append(time[curves[0] - stds[0] > 0.05].min())
high_times = [time[curve + std > 0.5].min() for curve, std in zip(curves, stds)]
high_times.append(time[curves[0] + stds[0] > 0.05].min())

time_diff = abs(times[0] - times[1])
print(time_diff)
time_error = [abs(low - high) / 2 for low, high in zip(low_times, high_times)]
time_error[2] = abs(times[2] - low_times[2])
time_err = float(np.sqrt(time_error[0] ** 2 + time_error[1] ** 2))
penetration_time = abs(times[2] - times[1])
penetration_time_err = float(np.sqrt(time_error[2] ** 2 + time_error[1] ** 2))

fig, axes = plt.subplots(1, 1)

for curve, std in zip(curves, stds):
    l = axes.plot(time, curve)
    axes.fill_between(time, curve - std, curve + std, color=l[0].get_c(), alpha=0.5)

for i in range(3):
    axes.axvline(times[i], color="green")
    axes.axvline(low_times[i], color="red", alpha=0.5)
    axes.axvline(high_times[i], color="red", alpha=0.5)

axes.set_title(
    f"Delta T_50 = {time_diff:.01f} +/- {time_err:.02f} min"
    + f"Penetration time = {penetration_time:.01f} +/- {penetration_time_err:.02f} min"
)

fig.savefig(snakemake.output.png)

inds = np.logical_not(np.any(np.isnan(heatmap), axis=0))
x_values = distance[inds]
max_dist = x_values.max()

with open(snakemake.output.yaml, "w") as f:
    yaml.dump(
        {
            "time in min": float(time_diff),
            "time err": time_err,
            "penetration time in min": float(penetration_time),
            "penetration time err in min": penetration_time_err,
            "max_dist": float(max_dist),
        },
        f,
    )
