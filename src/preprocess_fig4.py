import os
from typing import TYPE_CHECKING

import nd2
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from skimage.measure import regionprops
from tifffile import imread
from tqdm import trange

try:
    from snakemake.script import snakemake
except ImportError:
    if TYPE_CHECKING:
        from snakemake.iocontainers import snakemake  # noqa: TC004


VALUE_VARS = ["mean", "median", "75 %", "90 %", "95 %"]


def calc_ratios(values):
    means = np.concat(
        (
            np.mean(values, axis=0)[np.newaxis, :],
            np.percentile(values, q=[50, 75, 90, 95], axis=0),
        ),
        axis=0,
    )
    return np.concat((means[:, 1] / means[:, 0], means[:, 0], means[:, 1]))


def calc_df(img_, lbl):
    bg = np.median(img_[lbl == 0], axis=0)
    print(bg)
    img = img_ - bg

    objects = find_objects(lbl)

    values = []
    lbl_id = []
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        mask = lbl[sl] == i
        values.append(calc_ratios(img[sl][mask]))
        lbl_id.append(i)
    values_np = np.stack(values, axis=0)

    dfs = []
    for j, ty in enumerate(["ratio", "mScarlet", "gfp"]):
        values = {
            t: values_np[:, i + j * len(VALUE_VARS)] for i, t in enumerate(VALUE_VARS)
        }
        values["lbl_id"] = lbl_id
        values["type"] = ty
        dfs.append(pd.DataFrame(values))

    return pd.concat(dfs, ignore_index=True)


def get_props(lbl):
    objects = find_objects(lbl)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        mask = lbl[sl] == i
        if mask.sum() < 20:
            imprint = lbl[sl]
            imprint[mask] = 0
            lbl[sl] = imprint

    props = regionprops(lbl)
    values = {
        v: [p[v] for p in props]
        for v in [
            "label",
            # "eccentricity",
            "axis_major_length",
            "axis_minor_length",
            "area",
            # "solidity",
            # "perimeter",
        ]
    }
    return pd.DataFrame(values)


file = snakemake.wildcards.file
nd2_file = nd2.ND2File(snakemake.input.nd2)
nd2_dask = nd2_file.to_dask()

dfs = []
props_dfs = []
for t in range(2):
    for m in trange(3):
        img = nd2_dask[t, m, :, 1:].compute().transpose(0, 2, 3, 1)
        filename = f"{file}_t{t}_m{m}_mScarlet.tiff"
        lbl = imread(os.path.join("data/preprocessed/fig4", filename))
        lbl = lbl[: img.shape[0]]

        props_df = get_props(lbl)
        df = calc_df(img, lbl)

        for tmp in [props_df, df]:
            tmp["t"] = t
            tmp["m"] = m
            tmp["file"] = snakemake.wildcards.file

        props_dfs.append(props_df)
        dfs.append(df)


props_df = pd.concat(props_dfs, ignore_index=True)
props_df["aspect ratio"] = props_df["axis_major_length"] / props_df["axis_minor_length"]

df = pd.concat(dfs, ignore_index=True)
df = df.melt(
    id_vars=["t", "m", "file", "lbl_id", "type"],
    value_vars=VALUE_VARS,
    var_name="quantification",
)
df = df.pivot_table(
    columns="type", values="value", index=["t", "m", "file", "lbl_id", "quantification"]
).reset_index()
df = df[df["ratio"] > 0]
df = df[df["ratio"] < 10]

df = df.merge(
    props_df, left_on=["lbl_id", "t", "m", "file"], right_on=["label", "t", "m", "file"]
)

df.to_csv(snakemake.output.csv)
