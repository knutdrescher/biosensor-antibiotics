from snakemake.script import snakemake
import yaml
import matplotlib.pyplot as plt
import pandas as pd


times = []
errors = []
size = []

for y in snakemake.input.yamls:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        times.append(tmp["time in min"])
        errors.append(tmp["time err"])
        size.append(tmp["max_dist"])
for y in snakemake.input.HMBR:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        times.append(tmp["time in min"])
        errors.append(tmp["time err"])
        size.append(tmp["max_dist"])

fig, axes = plt.subplots(1, 1, figsize=[6, 5])

ax = axes
ax.errorbar(
    size[:-2],
    times[:-2],
    yerr=errors[:-2],
    linestyle="none",
    marker="x",
)
ax.errorbar(
    size[-2:],
    times[-2:],
    yerr=errors[-2:],
    linestyle="none",
    marker="x",
    label="HMBR control",
)
ax.set_xlabel("biofilm radius (µm)")
ax.set_ylabel("penetration time (min)")
ax.legend()

fig.savefig(snakemake.output.png)

df = pd.DataFrame(
    {
        "times": times[:-2],
        "errors": errors[:-2],
        "size": size[:-2],
    }
)
df.to_csv(snakemake.output.csv)
