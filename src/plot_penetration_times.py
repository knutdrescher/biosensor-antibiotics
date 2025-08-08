from snakemake.script import snakemake
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


times = ([], [])
errors = ([], [])
size = []
category = []

descrs = ["time in min", "penetration time in min"]
descrs_err = ["time err", "penetration time err in min"]

for y in snakemake.input.yamls:
    with open(y) as f:
        tmp = yaml.safe_load(f)
        for i in range(2):
            times[i].append(tmp[descrs[i]])
            errors[i].append(tmp[descrs_err[i]])
        size.append(tmp["max_dist"])
        category.append("normal")
# for y in snakemake.input.HMBR:
#     with open(y) as f:
#         tmp = yaml.safe_load(f)
#         times.append(tmp["time in min"])
#         errors.append(tmp["time err"])
#         size.append(tmp["max_dist"])
#         category.append("HMBR")

fig, axes = plt.subplots(1, 2, figsize=[9, 5])

for i in range(2):
    ax = axes[i]
    x = size
    y = times[i]
    ax.errorbar(
        x,
        y,
        yerr=errors[i],
        linestyle="none",
        marker="x",
    )
    # ax.errorbar(
    #    size[-2:],
    #    times[-2:],
    #    yerr=errors[-2:],
    #    linestyle="none",
    #    marker="x",
    #    label="HMBR control",
    # )
    ax.set_xlabel("biofilm radius (µm)")
    ax.set_ylabel("penetration time (min)")
    # ax.legend()

    res = spearmanr(x, y, alternative="greater")
    print(res)
    ax.set_title(f"Spearman r = {res.statistic:.02f}\n pvalue = {res.pvalue:.02e}")


fig.savefig(snakemake.output.png)

df = pd.DataFrame(
    {
        "times": times[0],
        "errors": errors[0],
        "times penetration": times[1],
        "errors penetration": errors[1],
        "size": size,
        "category": category,
    }
)
df.to_csv(snakemake.output.csv)
