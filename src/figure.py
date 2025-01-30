import matplotlib.pyplot as plt
import nd2
import yaml
import pandas as pd
import numpy as np
from snakemake.script import snakemake
from cmap import Colormap
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from tifffile import imread
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import ConnectionPatch
from scipy import stats
from matplotlib.transforms import blended_transform_factory

# plt.rc("font", family="Helvetica")


fig = plt.figure(figsize=[18, 16], constrained_layout=True)

img_ratio = 1.2
gs = fig.add_gridspec(
    7,
    11,
    width_ratios=[
        0.07,
        img_ratio,
        img_ratio,
        img_ratio,
        0.1,
        img_ratio,
        0.08,
        1,
        0.06,
        1,
        0.1,
    ],
    height_ratios=[0.05, 1, 1, 1, 1, 0.75, 0.75],
    # hspace=0,
    # wspace=0,
)

top_axes = [[] for i in range(4)]
top_axes.append([None for _ in range(5)])
top_axes.append([None for _ in range(5)])

for i in range(6):
    for j_, j in enumerate(
        list(range(1, 4)) + list(range(5, 8)) + list(range(9, 11)) + [8]
    ):
        if i in [4, 5] and j_ not in [5, 6, 7, 8]:
            continue
        top_axes[i].append(fig.add_subplot(gs[i + 1, j]))

print(top_axes)

# control_label_ax = fig.add_subplot(gs[5:, 6])


def img_axis(ax):
    ax.axis("off")


cyan_cmap = Colormap("cmap:cyan").to_mpl()
yellow_cmap = Colormap("cmap:yellow").to_mpl()
red_crop_position = [
    (503, 181),
    (224, 274),
    (247, 248),
    (0, 0),
]
sizes_ylabel = ["50 µm", "100 µm", "200 µm", "400 µm"]
time_cutoffs = [4, 5, 8, 8, 8, 60]
distance_boundary_label = "Distance to biofilm boundary (µm)"


def make_square(x):
    size = min(x.shape)
    half_size = size // 2
    center_y = x.shape[0] // 2
    center_x = x.shape[1] // 2

    start_y = center_y - half_size
    sly = slice(start_y, start_y + size)
    start_x = center_x - half_size
    slx = slice(start_x, start_x + size)

    return x[sly, slx]


for i in range(len(snakemake.input.nd2_gfp)):
    print(i)
    file = snakemake.input.nd2_gfp[i]
    with nd2.ND2File(file) as nd2_file:
        dask = nd2_file.to_dask()

        times = []
        for j in range(nd2_file.metadata.contents.frameCount):
            times.append(
                nd2_file.frame_metadata(j).channels[0].time.relativeTimeMs
            )
        exp_times = np.array(times) / 1000 / 60
        gfp_time = np.argwhere(exp_times < 8).max()

        img0 = make_square(dask[0, 0, ...].compute())
        img1 = make_square(dask[gfp_time, 0, ...].compute())
        pixel_size = nd2_file.metadata.channels[0].volume.axesCalibration[0]
        imgs = [img0, img1]
        vmin = min(img.min() for img in imgs)
        vmax = max(img.max() for img in imgs)

    for j, img in enumerate([img0, img1]):
        ax = top_axes[i][j + 1]
        ax.imshow(img, cmap=cyan_cmap, vmin=vmin, vmax=vmax)
        scalebar = ScaleBar(
            pixel_size,
            "µm",
            length_fraction=0.2,
            location="lower right",
            color="white",
            box_alpha=0.0,
            scale_loc="top",
        )
        if j == 1:
            ax.add_artist(scalebar)
        if i != 0:
            img_axis(ax)

    with nd2.ND2File(snakemake.input.nd2_red[i]) as nd2_red:
        y, x = red_crop_position[i]
        sly = slice(y, y + imgs[0].shape[0])
        slx = slice(x, x + imgs[0].shape[1])
        red_img = make_square(nd2_red.asarray()[1, sly, slx])
        ax = top_axes[i][0]
        ax.imshow(red_img, cmap=yellow_cmap)
        # img_axis(ax)
        if i != 0:
            ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel(f"~ {sizes_ylabel[i]}", color="black")

    edt_file = snakemake.input.edt[i]
    edt = make_square(imread(edt_file, key=0))

    ax = top_axes[i][3]
    cax = top_axes[i][4]
    im = ax.imshow(edt, cmap="gray")
    img_axis(ax)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel(distance_boundary_label)

for i in range(6):
    heatmap = np.load(snakemake.input.heatmaps[i])
    deriv = np.load(snakemake.input.derivs[i])
    x = np.load(snakemake.input.xs[i])
    t = np.load(snakemake.input.ts[i])
    deriv_t = np.load(snakemake.input.deriv_ts[i])

    time_cutoff_ind = np.argwhere(t < time_cutoffs[i]).max()
    time_cutoff_ind_deriv = np.argwhere(deriv_t < time_cutoffs[i]).max()
    t = t[:time_cutoff_ind]
    deriv_t = deriv_t[:time_cutoff_ind_deriv]
    heatmap = heatmap[:time_cutoff_ind]
    inds = np.logical_not(np.isnan(heatmap).all(axis=0))
    heatmap = heatmap[:, inds]
    x_heatmap = x[inds]
    deriv = deriv[:time_cutoff_ind_deriv]
    inds = np.logical_not(np.isnan(deriv).all(axis=0))
    deriv = deriv[:, inds]
    x_deriv = x[inds]

    ax = top_axes[i][5]
    X, Y = np.meshgrid(t, x_heatmap)

    im = ax.pcolormesh(
        X, Y, heatmap.T, shading="nearest", cmap="inferno", rasterized=True
    )
    ax.set_xlabel("Time (min)")
    ax.set_facecolor("gray")
    if i < 4:
        ax.set_ylabel(distance_boundary_label)

    cax = top_axes[i][8]
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("TMP-FAST Intensity (a.u.)")
    cbar.ax.set_yticks([])

    ax = top_axes[i][6]
    X, Y = np.meshgrid(deriv_t, x_deriv)

    norm = TwoSlopeNorm(vcenter=0.0)
    im = ax.pcolormesh(
        X,
        Y,
        deriv.T,
        shading="nearest",
        cmap="seismic",
        norm=norm,
        rasterized=True,
    )
    ax.set_xlabel("Time (min)")
    ax.set_facecolor("gray")

    cax = top_axes[i][7]
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("d Intensity/dt (a.u./s)")

    if i < 4:
        top_axes[i][5].sharey(top_axes[i][6])
        top_axes[i][4].sharey(top_axes[i][6])

# control_label_ax.set_ylabel(distance_boundary_label)
# control_label_ax.yaxis.set_label_position("right")
# control_label_ax.yaxis.tick_right()

# new_trafo = blended_transform_factory(
#     x_transform=top_axes[4][5].yaxis.label.get_transform(),
#     y_transform=top_axes[4][5].transAxes,
# )
# new_distance_label = distance_boundary_label.replace(" biofilm ", " ")
# top_axes[4][5].set_ylabel(new_distance_label)
top_axes[4][5].set_ylabel(distance_boundary_label, y=-0.2)
# top_axes[5][5].set_ylabel("")
fig.align_ylabels([top_axes[i][5] for i in range(6)])
# fig.align_ylabels([top_axes[i][5] for i in range(4)])
# top_axes[3][4].set_ylabel(distance_boundary_label, clip_on=False, y=-1)

# fig.text(
#    -0.2,
#    -0.1,
#    distance_boundary_label,
#    transform=top_axes[4][5].transAxes,
#    ha="right",
#    va="center",
#    rotation=90,
# )

for i, label in zip([4, 5], ["pFAST control", "HMBR control"]):
    top_axes[i][5].set_title(label, loc="left")
    # fig.text(
    #     -0.25,
    #     1.02,
    #     label,
    #     transform=top_axes[i][5].transAxes,
    #     ha="left",
    #     va="bottom",
    # )
# img_axis(control_label_ax)


xpos = 0.05
ypos = 0.92
ax = top_axes[0][0]
ax.text(xpos, ypos, "mScarlet", transform=ax.transAxes, color="yellow")


def add_xy_arrow(
    ax,
    xarr=0.04,
    yarr=0.04,
    dx=0.10,
    dy=0.10,
    offset_x=0.07,
    offset_y=0.09,
    head_width=0.03,
):
    ax.arrow(
        xarr,
        yarr,
        dx,
        0,
        transform=ax.transAxes,
        color="white",
        head_width=head_width,
    )
    ax.text(
        xarr + dx + offset_x,
        yarr,
        "x",
        color="white",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.arrow(
        xarr,
        yarr,
        0,
        dy,
        transform=ax.transAxes,
        color="white",
        head_width=head_width,
    )
    ax.text(
        xarr,
        yarr + dy + offset_y,
        "y",
        color="white",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


add_xy_arrow(ax)
ax = top_axes[0][1]
ax.text(xpos, ypos, "TMP-FAST", transform=ax.transAxes, color="cyan")
ax = top_axes[0][3]
ax.text(xpos, ypos, "Distance", transform=ax.transAxes, color="white")
add_xy_arrow(ax)

for j, label in enumerate(["t = -5 min", "t = -5 min", "t = +8 min"]):
    ax = top_axes[0][j]
    if j != 0:
        ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_xlabel(label)
    ax.xaxis.set_label_position("top")

diameter_arrow_ax = fig.add_subplot(gs[1:5, 0])
diameter_arrow_ax.annotate(
    "",
    (0.5, 0.01),
    (0.5, 0.99),
    xycoords=diameter_arrow_ax.transAxes,
    textcoords=diameter_arrow_ax.transAxes,
    arrowprops=dict(facecolor="black", shrink=0.05),
)
diameter_arrow_ax.set_ylabel("Biofilm diameter")
diameter_arrow_ax.get_xaxis().set_visible(False)
diameter_arrow_ax.set_yticks([])
for i in ["top", "right", "left", "bottom"]:
    diameter_arrow_ax.spines[i].set_visible(False)

ax = top_axes[0][5]
x_start = -0.18
offset_periphery = 0.02
offset_center = 0.005
periphery = -0.05
center = 1.005
arrow = ConnectionPatch(
    (x_start, periphery + offset_periphery),
    (x_start, center - offset_center),
    coordsA=ax.transAxes,
    coordsB=ax.transAxes,
    arrowstyle="->",
    color="black",
    annotation_clip=False,
    lw=2,
)
fig.text(x_start, periphery, "Periphery", transform=ax.transAxes, va="top")
fig.text(x_start, center, "Center", transform=ax.transAxes, va="bottom")
fig.add_artist(arrow)


subfigs = fig.add_subfigure(gs[5:, :6])

ax = subfigs.add_subplot(1, 2, 1)
df = pd.read_csv(snakemake.input.times)
ax.errorbar(
    2 * df["size"],
    df["times"],
    yerr=df["errors"],
    marker="o",
    ls="none",
)
ax.set_xlabel("Biofilm diameter (µm)")
ax.set_ylabel("Penetration time (min)")
x = df["size"]
y = df["times"]


def statistic(x):  # permute only `x`
    return stats.spearmanr(x, y, alternative="greater").statistic


res_exact = stats.permutation_test((x,), statistic, permutation_type="pairings")
print(res_exact)
ax.text(
    0.05,
    0.9,
    f"Spearman p-value: {res_exact.pvalue}",
    transform=ax.transAxes,
)

ax = subfigs.add_subplot(1, 2, 2)
df = pd.read_csv(snakemake.input.slopes)
ax.errorbar(
    2 * df["size"],
    df["speed"],
    yerr=df["speed_err"],
    marker="o",
    ls="none",
)
ax.set_xlabel("Biofilm diameter (µm)")
ax.set_ylabel("Penetration speed (µm/sec)")

x = df["size"]
y = df["speed"]


def statistic(x):  # permute only `x`
    return stats.spearmanr(x, y, alternative="less").statistic


res_exact = stats.permutation_test((x,), statistic, permutation_type="pairings")
print(res_exact)
ax.text(
    0.95,
    0.9,
    f"Spearman p-value: {res_exact.pvalue}",
    transform=ax.transAxes,
    ha="right",
)


fig.savefig(snakemake.output.png)
fig.savefig(snakemake.output.svg, dpi=300)
fig.savefig(snakemake.output.pdf)
