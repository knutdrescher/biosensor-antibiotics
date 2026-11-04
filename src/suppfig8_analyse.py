from typing import TYPE_CHECKING

from tifffile import imread

from suppfig8_helpers import (
    calc_assignment,
    get_melt_df,
    open_file,
    reassign_lbls,
)

try:
    from snakemake.script import snakemake
except ImportError:
    if TYPE_CHECKING:
        from snakemake.iocontainers import snakemake  # noqa: TC004
file = snakemake.wildcards.file
pos = int(snakemake.wildcards.pos)

img_da, nd2_files = open_file(file)
img = img_da[:, pos, :, 1:].compute()

lbl0 = imread(snakemake.input.lbls[0])
lbl1 = imread(snakemake.input.lbls[1])

aligned_lbls = imread(snakemake.input.aligned_lbls)

assignment = calc_assignment(aligned_lbls[0], aligned_lbls[1])
reassigned_lbl1 = reassign_lbls(lbl1, assignment)

df = get_melt_df(img.transpose(0, 1, 3, 4, 2), lbl0, reassigned_lbl1)
df["file"] = file
df["pos"] = pos

df.to_csv(snakemake.output.csv)

for nd2_file in nd2_files:
    nd2_file.close()
