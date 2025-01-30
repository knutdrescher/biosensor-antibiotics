import yaml
from snakemake.script import snakemake
import numpy as np
import matplotlib.pyplot as plt


SMOOTHING_N = 10


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


heatmap = np.load(snakemake.input.heatmap)
time = np.load(snakemake.input.time)
distance = np.load(snakemake.input.x)

with open(snakemake.input.yaml) as f:
    start_stop = yaml.safe_load(f)

end_time_cutoff = 10
if "HMBR" in snakemake.wildcards.file:
    end_time_cutoff = 30
time_cutoff = np.argwhere(time < end_time_cutoff).max()

time = time[:time_cutoff]
heatmap = heatmap[:time_cutoff, :]

t_max = np.nanmax(heatmap, axis=1)

t_max_smoothened = running_mean(t_max, N=SMOOTHING_N)
time_smoothened = running_mean(time, N=SMOOTHING_N)

fig, axes = plt.subplots(1, 2, figsize=[8, 6])

ax = axes[0]

ax.plot(time_smoothened, t_max_smoothened)

for i in range(2):
    ax.axvline(start_stop["start"][i], color="green")
    ax.axvline(start_stop["stop"][i], color="red")

fig.savefig(snakemake.output.png)

start = np.mean(start_stop["start"])
stop = np.mean(start_stop["stop"])

std1 = np.std(start_stop["start"])
std2 = np.std(start_stop["stop"])

inds = np.logical_not(np.any(np.isnan(heatmap), axis=0))
x_values = distance[inds]
max_dist = x_values.max()

with open(snakemake.output.yaml, "w") as f:
    yaml.dump(
        {
            "time in min": float(stop - start),
            "time err": float(np.sqrt(std1**2 + std2**2)),
            "max_dist": float(max_dist),
        },
        f,
    )
