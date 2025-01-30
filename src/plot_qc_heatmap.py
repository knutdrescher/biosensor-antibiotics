import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from snakemake.script import snakemake


heatmap_mean = np.load(snakemake.input.i_mean)
heatmap_std = np.load(snakemake.input.i_std)
deriv_mean = np.load(snakemake.input.deriv_mean)
deriv_std = np.load(snakemake.input.deriv_std)

exp_times = np.load(snakemake.input.exp_times)
deriv_time = np.load(snakemake.input.deriv_time)
x = np.load(snakemake.input.x)

fig, axes = plt.subplots(2, 2, figsize=[12, 12])

end_time_cutoff = 8
if "HMBR" in snakemake.wildcards.file:
    end_time_cutoff = 60
time_cutoff = np.argwhere(exp_times < end_time_cutoff).max()
time_cutoff_deriv = np.argwhere(deriv_time < end_time_cutoff).max()

X, Y = np.meshgrid(x, exp_times[:time_cutoff])
X_t_deriv, Y_t_deriv = np.meshgrid(x, deriv_time[:time_cutoff_deriv])

signal = ["mean GFP signal", "std GFP signal"]
signal_deriv = ["time derivative of mean GFP signal", "std of time derivative"]
x_label = "distance to biofilm boundary (µm)"
time_label = "time (min)"

for i, heatmap in enumerate([heatmap_mean, heatmap_std]):
    ax = axes[0, i]
    ax.invert_yaxis()
    im = ax.pcolormesh(
        X,
        Y,
        heatmap[:time_cutoff, :],
        shading="nearest",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(signal[i])

    ax.set_ylabel(time_label)
    ax.set_xlabel(x_label)

    ax.set_aspect("auto")

for i, heatmap in enumerate([deriv_mean, deriv_std]):
    ax = axes[1, i]
    ax.invert_yaxis()
    im = ax.pcolormesh(
        X_t_deriv,
        Y_t_deriv,
        heatmap[:time_cutoff_deriv, :],
        shading="nearest",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(signal_deriv[i])

    ax.set_ylabel(time_label)
    ax.set_xlabel(x_label)

    ax.set_aspect("auto")

fig.savefig(snakemake.output.png)
