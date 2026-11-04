import os

import dask.array as da
import nd2
import numpy as np
import pandas as pd
from numpy.fft import fftshift, irfftn, rfftn
from scipy.ndimage import find_objects
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops


def open_file(file):
    path = "data/raw/suppfig8"
    if file == "NCCRBM128_sample4":
        files = [
            "NCCRBM128_sample4_before.nd2",
            "NCCRBM128_sample4_TMP.nd2",
        ]

        imgs = [nd2.ND2File(os.path.join(path, file)) for file in files]

        return da.stack([img.to_dask() for img in imgs], axis=0), imgs
    else:
        nd2_file = nd2.ND2File(os.path.join(path, f"{file}.nd2"))
        return nd2_file.to_dask(), [nd2_file]


def fft_translation(img0, img1):
    fft_img0 = rfftn(img0)
    fft_img1 = rfftn(img1)
    fft_img1_cc = np.conjugate(fft_img1)

    mult = fft_img0 * fft_img1_cc
    mult /= np.abs(mult)
    inverse = irfftn(mult, s=img0.shape, axes=range(img0.ndim))

    inverse_shifted = fftshift(inverse)
    # Offset relative to the center of the array
    peak = np.array(np.unravel_index(np.argmax(inverse_shifted), shape=img0.shape))
    offset = peak - np.array(img0.shape) // 2

    return offset


def create_pads(offset, img_shape):
    pads0 = [(0, 0)]
    pads1 = [(0, 0)]
    for i, o in enumerate(offset):
        if o >= 0:
            pads0.append((0, o))
            pads1.append((o, 0))
        else:
            pads1.append((0, -o))
            pads0.append((-o, 0))

    return pads0, pads1


def calc_assignment(lbl0, lbl1):
    cost_matrix = np.zeros((lbl0.max(), lbl1.max()), dtype=int)
    inds = np.logical_and(lbl0 > 0, lbl1 > 0)

    for i, j in zip(lbl0[inds] - 1, lbl1[inds] - 1):
        cost_matrix[i, j] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    return {
        int(k + 1): int(v + 1)
        for k, v in zip(col_ind, row_ind)
        if cost_matrix[v, k] > 10
    }


def reassign_lbls(lbl, assignment):
    new_lbl = np.zeros_like(lbl)
    for i, sl in enumerate(find_objects(lbl), 1):
        if sl is None:
            continue
        if i not in assignment:
            continue
        mask = lbl[sl] == i
        new_tmp = np.copy(new_lbl[sl])

        new_tmp[mask] = assignment[i]
        new_lbl[sl] = new_tmp
    return new_lbl


def get_props(lbl):
    objects = find_objects(lbl)

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        mask = lbl[sl] == i
        if mask.sum() < 30:
            imprint = lbl[sl]
            imprint[mask] = 0
            lbl[sl] = imprint

    props = regionprops(lbl)
    parameters = [
        "label",
        # "eccentricity",
        "axis_major_length",
        "axis_minor_length",
        "area",
        # "solidity",
        # "perimeter",
    ]
    values = {v: [] for v in parameters}
    for p in props:
        try:
            tmp = {v: p[v] for v in parameters}
        except ValueError:
            print("Skipping lbl_id", p["label"])
            continue
        for v in parameters:
            values[v].append(tmp[v])

    df = pd.DataFrame(values)
    df["aspect ratio"] = df["axis_major_length"] / df["axis_minor_length"]
    return df


def get_melt_df(img, lbl0, lbl1):
    df0 = calc_df(img[0], lbl0)
    df1 = calc_df(img[1], lbl1)

    props_df0 = get_props(lbl0)
    props_df1 = get_props(lbl1)
    props_df = props_df0.merge(
        props_df1, left_on="label", right_on="label", suffixes=("_0", "_1")
    )

    dfs = []
    for i in ["ratio", "gfp", "mScarlet"]:
        tmp0 = df0.loc[df0["type"] == i, ["median", "lbl_id"]]
        tmp1 = df1.loc[df1["type"] == i, ["median", "lbl_id"]]

        merge = tmp0.merge(
            tmp1, left_on="lbl_id", right_on="lbl_id", suffixes=("_0", "_1")
        )

        merge = merge[merge["median_1"] > 0]
        merge = merge[merge["median_0"] > 0]

        merge["ratio"] = merge["median_1"] / merge["median_0"]
        merge["diff"] = merge["median_1"] - merge["median_0"]

        dfs.append(merge)

    df = dfs[0].merge(
        dfs[1], left_on="lbl_id", right_on="lbl_id", suffixes=("_ratio", "_gfp")
    )
    df = df.merge(dfs[2], left_on="lbl_id", right_on="lbl_id")
    df = df.merge(props_df, left_on="lbl_id", right_on="label")

    df = df[(df["aspect ratio_0"] > 2) & (df["aspect ratio_1"] > 2)]

    return df


def calc_ratios(values):
    means = np.concat(
        (
            np.mean(values, axis=0)[np.newaxis, :],
            np.percentile(values, q=[50, 75, 90, 95], axis=0),
        ),
        axis=0,
    )
    return np.concat((means[:, 1] / means[:, 0], means[:, 0], means[:, 1]))


VALUE_VARS = ["mean", "median", "75 %", "90 %", "95 %"]


def calc_df(img, lbl):
    lbl = lbl[: img.shape[0]]
    bg = np.median(img[lbl == 0], axis=0)
    img = np.clip(img - bg, 1, None)

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
