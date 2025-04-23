from snakemake.script import snakemake
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd


slopes = []
errors = []
size = []
category = []

for y in snakemake.input.yamls:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        if tmp["slope_err"] > 0.1 / 60:
            print(y)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])
        category.append("normal")

start_HMBR = len(slopes)
end_HMBR = start_HMBR + len(snakemake.input.HMBR)
for y in snakemake.input.HMBR:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])
        category.append("HMBR")

start_pH6 = len(slopes)
end_pH6 = start_pH6 + len(snakemake.input.pH6)
for y in snakemake.input.pH6:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])
        category.append("pH 6")

start_pH8 = len(slopes)
end_pH8 = start_pH8 + len(snakemake.input.pH8)
for y in snakemake.input.pH8:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])
        category.append("pH 8")

start_pH85 = len(slopes)
end_pH85 = start_pH85 + len(snakemake.input.pH85)
for y in snakemake.input.pH85:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])
        category.append("pH 8.5")

start_GFP = len(slopes)
end_GFP = start_GFP + len(snakemake.input.GFP)
for y in snakemake.input.GFP:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        slopes.append(tmp["slope"])
        errors.append(tmp["slope_err"])
        size.append(tmp["max_dist"])
        category.append("GFP")


slopes = np.array(slopes) * 60
errors = np.array(errors) * 60

fig, axes = plt.subplots(1, 2, figsize=[8, 5])

ax = axes[0]
labels = [None, "HMBR control", "pH 6.0", "pH 8.0", "pH 8.5", "TMP-GFP"]
sl_normal = slice(0, start_HMBR)
sl_HMBR = slice(start_HMBR, end_HMBR)
sl_pH6 = slice(start_pH6, end_pH6)
sl_pH8 = slice(start_pH8, end_pH8)
sl_pH85 = slice(start_pH85, end_pH85)
sl_GFP = slice(start_GFP, end_GFP)
for sl, label in zip(
    [sl_normal, sl_HMBR, sl_pH6, sl_pH8, sl_pH85, sl_GFP], labels
):
    ax.errorbar(
        size[sl],
        slopes[sl],
        yerr=errors[sl],
        linestyle="none",
        marker="x",
        label=label,
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
        "slopes": slopes,
        "errors": errors,
        "size": size,
        "speed": speeds,
        "speed_err": speed_err,
        "category": category,
    }
)
df.to_csv(snakemake.output.csv)
