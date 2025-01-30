from snakemake.script import snakemake
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd


slopes = []
errors = []
size = []

for y in snakemake.input.yamls:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        if tmp["slope_err"] > 0.1 / 60:
            print(y)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])

for y in snakemake.input.HMBR:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])


slopes = np.array(slopes) * 60
errors = np.array(errors) * 60

fig, axes = plt.subplots(1, 2, figsize=[8, 5])

ax = axes[0]
ax.errorbar(
    size[:-2],
    slopes[:-2],
    yerr=errors[:-2],
    linestyle="none",
    marker="x",
)
ax.errorbar(
    size[-2:],
    slopes[-2:],
    yerr=errors[-2:],
    linestyle="none",
    marker="x",
    label="HMBR control",
)
ax.set_xlabel("biofilm radius (µm)")
ax.set_ylabel("penetration slope in sec/µm")
ax.legend()

speeds = 1 / slopes
speed_err = np.abs(errors / slopes**2)

ax = axes[1]
ax.errorbar(
    size[:-2],
    speeds[:-2],
    yerr=speed_err[:-2],
    linestyle="none",
    marker="x",
)
ax.errorbar(
    size[-2:],
    speeds[-2:],
    yerr=speed_err[-2:],
    linestyle="none",
    marker="x",
    label="HMBR control",
)
ax.set_xlabel("biofilm radius (µm)")
ax.set_ylabel("penetration speed (1/slope) in µm/sec")
ax.legend()

fig.savefig(snakemake.output.png)

df = pd.DataFrame(
    {
        "slopes": slopes[:-2],
        "errors": errors[:-2],
        "size": size[:-2],
        "speed": speeds[:-2],
        "speed_err": speed_err[:-2],
    }
)
df.to_csv(snakemake.output.csv)
