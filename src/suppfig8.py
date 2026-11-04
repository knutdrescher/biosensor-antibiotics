from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import iqr

try:
    from snakemake.script import snakemake
except ImportError:
    if TYPE_CHECKING:
        from snakemake.iocontainers import snakemake  # noqa: TC004


def plot_stats(df, vals):
    for val in vals:
        print(val, "median:", np.median(df[val]), "iqr:", iqr(df[val]))
        print(val, "mean:", np.mean(df[val]), "std:", np.std(df[val]))


dfs = [pd.read_csv(f, index_col=0) for f in snakemake.input.csvs]

all_df = pd.concat(dfs, ignore_index=True)
all_df["area ratio"] = all_df["area_1"] / all_df["area_0"]

df_melt = all_df.melt(
    id_vars=["lbl_id", "file", "pos"], value_vars=["ratio_ratio", "ratio_gfp", "ratio"]
)

tmp_df = all_df[(0.5 < all_df["area ratio"]) & (all_df["area ratio"] < 2)]

df_melt2 = tmp_df.melt(
    id_vars=["lbl_id", "file", "pos"], value_vars=["median_0_ratio", "median_1_ratio"]
)

sns.violinplot(
    df_melt2,
    x="variable",
    y="value",
)

with open(snakemake.output.txt, "w") as f, redirect_stdout(f):
    plot_stats(tmp_df, ["median_0_ratio", "median_1_ratio"])

plt.savefig(snakemake.output.pdf)
df_melt2.to_excel(snakemake.output.xlsx)
