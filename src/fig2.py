from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from statannotations.Annotator import Annotator

try:
    from snakemake.script import snakemake
except ImportError:
    from snakemake.iocontainers import snakemake

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
    }
)
ytick_fontsize = 6

root_path = snakemake.input.xlsx

triton = {
    "woTriton": "Without Triton",
    "0.2%Triton": "0.2 % Triton",
}
triton_order = [v for v in triton.values()]

antibiotic = {
    "TetFABS": "Doxy",
    "TrimFABS": "TMP",
}


def get_df(path, sheet_name):
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
    )

    for col in df.columns:
        if col == "Time, min":
            continue
        mi = df[col].min()
        ma = df[col].max()
        df[col] = (df[col] - mi) / (ma - mi)

    df = df.melt(id_vars="Time, min").dropna()

    df["split"] = df["variable"].str.split("_")
    df["Antibiotic"] = df["split"].apply(lambda x: antibiotic[str(x[0])])
    df["Condition"] = df["split"].apply(lambda x: triton[str(x[1])])
    df["Well"] = df["split"].apply(lambda x: int(x[2][4:]))
    try:
        df["Position"] = df["split"].apply(lambda x: int(x[3][3:]))
    except IndexError:
        df["Position"] = 1

    return df


abx_df = pd.concat(
    [
        get_df(root_path, s)
        for s in ["Fig.2c,d_TrimFABS_Macrophages", "Fig.2c,d_TetFABS_Macrophages"]
    ],
    ignore_index=True,
)
abx_df["Macrophages"] = "With"
control_df = get_df(root_path, "Fig.2d_Bacterial_susp")
control_df["Macrophages"] = "Without"
control_df["Time, min"] += 2
df = pd.concat([abx_df, control_df], ignore_index=True)

means = (
    df.groupby(
        ["Time, min", "Antibiotic", "Condition", "Macrophages", "Well", "Position"]
    )["value"]
    .mean()
    .reset_index()
)


inds = means["value"] > 0.5
time = "$T_{50}$ in min"
time = "T\u2085\u2080, min"
times = (
    means[inds]
    .groupby(["Antibiotic", "Condition", "Macrophages", "Well", "Position"])[
        "Time, min"
    ]
    .min()
    - 2
).reset_index()
times = times.rename({"Time, min": time}, axis=1)

mean_times = times.groupby(["Antibiotic", "Condition", "Macrophages"])[time].mean()
mean_times.name = f"mean of {time}"
stds_times = times.groupby(["Antibiotic", "Condition", "Macrophages"])[time].std()
stds_times.name = f"std of {time}"

pd.concat([mean_times, stds_times], axis=1).to_excel(snakemake.output.xlsx)


x = "Macrophages"
col = "Antibiotic"
hue = "Condition"
order = ["Tet", "Doxy", "TMP"]
order = ["With", "Without"]
col_order = ["Doxy", "TMP"]

plt.figure(dpi=600, figsize=[3, 4])


fg = sns.catplot(
    times,
    x=x,
    y=time,
    hue=hue,
    order=order,
    hue_order=triton_order,
    col=col,
    col_order=col_order,
    kind="bar",
    height=4,
    aspect=0.6,
)

fg.map(
    sns.stripplot,
    x,
    time,
    hue,
    dodge=True,
    palette="dark:black",
    order=order,
    hue_order=triton_order,
    legend=False,
)

for i, abx in enumerate(col_order):
    ax = fg.axes[0, i]

    pairs = [
        (("With", triton_order[0]), ("Without", triton_order[0])),
        (("With", triton_order[1]), ("Without", triton_order[1])),
        (("With", triton_order[0]), ("With", triton_order[1])),
        (("Without", triton_order[0]), ("Without", triton_order[1])),
    ]

    tmp_times = times[times[col] == abx]
    annotator = Annotator(
        ax,
        pairs,
        data=tmp_times,
        x=x,
        y=time,
        hue=hue,
        order=order,
        hue_order=triton_order,
    )
    annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")

    with open(snakemake.output.pvalue, "a") as f, redirect_stdout(f):
        print("*** ", abx, " ***")
        annotator.apply_and_annotate()
        print("\n")

    # ax.set_xlabel("")
    # if i == 0:
    #   ax.legend().set_title(None)

    ytick_font = font_manager.FontProperties(
        family="Helvetica Neue", size=ytick_fontsize
    )

    # Anwenden auf yticks
    for label in ax.get_yticklabels():
        label.set_fontproperties(ytick_font)

plt.tight_layout()
plt.savefig(snakemake.output.panel_d)

plt.figure(dpi=600, figsize=[5, 4])

sns.relplot(
    means,
    x="Time, min",
    y="value",
    hue="Condition",
    hue_order=triton_order,
    col="Antibiotic",
    row="Macrophages",
    kind="line",
)

plt.tight_layout()
plt.savefig(snakemake.output.panel_c)
