from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from snakemake.script import snakemake
except ImportError:
    if TYPE_CHECKING:
        from snakemake.iocontainers import snakemake  # noqa: TC004

dfs = []
for csv in snakemake.input.csvs:
    dfs.append(pd.read_csv(csv, index_col=0))

df = pd.concat(dfs, ignore_index=True)

df["reps"] = df["file"] + df["m"].astype(str)

sns.catplot(
    data=df[df["aspect ratio"] > 2],
    x="t",
    y="ratio",
    col="quantification",
    row="reps",
    kind="violin",
)

plt.savefig(snakemake.output.svg_all)

sns.catplot(
    data=df[df["aspect ratio"] > 2],
    x="t",
    y="ratio",
    col="quantification",
    kind="violin",
)

plt.savefig(snakemake.output.svg)

df[(df["aspect ratio"] > 2) & (df["quantification"] == "median")][
    [
        "t",
        "gfp",
        "mScarlet",
        "ratio",
        "label",
        "axis_major_length",
        "axis_minor_length",
        "area",
        "aspect ratio",
    ]
].to_excel(snakemake.output.xlsx_all)

df_panel_h = df[
    (df["m"] == 0)
    & (df["file"] == "Fig.4_h_TrimFABS_ANOX_mScarlet_insert1")
    & (df["aspect ratio"] > 2)
    & (df["quantification"] == "median")
]
sns.catplot(
    data=df_panel_h,
    x="t",
    y="ratio",
    kind="violin",
)

plt.savefig(snakemake.output.pdf)

df_panel_h[
    [
        "t",
        "gfp",
        "mScarlet",
        "ratio",
        "label",
        "axis_major_length",
        "axis_minor_length",
        "area",
        "aspect ratio",
    ]
].to_excel(snakemake.output.xlsx)
