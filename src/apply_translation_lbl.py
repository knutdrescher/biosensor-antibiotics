from typing import TYPE_CHECKING

import numpy as np
from tifffile import imread, imwrite

from suppfig8_helpers import create_pads

try:
    from snakemake.script import snakemake
except ImportError:
    if TYPE_CHECKING:
        from snakemake.iocontainers import snakemake  # noqa: TC004
file = snakemake.wildcards.file
pos = int(snakemake.wildcards.pos)

offset = np.load(snakemake.input.npy)

lbls = [imread(f) for f in snakemake.input.lbls]
shape = lbls[0].shape
shape = (shape[0], 0, shape[1], shape[2])
pads0, pads1 = create_pads(offset, shape)

x0 = np.pad(lbls[0], pads0[1:], mode="constant")
x1 = np.pad(lbls[1], pads1[1:], mode="constant")

imwrite(snakemake.output.tif, np.stack([x0, x1], axis=0))
