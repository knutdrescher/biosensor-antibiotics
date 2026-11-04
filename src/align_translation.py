from typing import TYPE_CHECKING

import numpy as np

from suppfig8_helpers import fft_translation, open_file

try:
    from snakemake.script import snakemake
except ImportError:
    if TYPE_CHECKING:
        from snakemake.iocontainers import snakemake  # noqa: TC004


file = snakemake.wildcards.file
img_da, nd2_files = open_file(file)
pos = int(snakemake.wildcards.pos)

img0 = img_da[0, pos].compute()
img1 = img_da[1, pos].compute()

if "sample5" in file and pos == 3:
    offset = fft_translation(img0[11:26, 0], img1[:15, 0])
    offset[0] += 11
else:
    offset = fft_translation(img0[:, 0], img1[:, 0])

np.save(snakemake.output.npy, offset)

for nd2_file in nd2_files:
    nd2_file.close()
