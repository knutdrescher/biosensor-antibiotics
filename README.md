# Monitoring intracellular antibiotic concentrations in real-time using allosteric biosensors

Code repository for

Monitoring intracellular antibiotic concentrations in real-time using allosteric biosensors

Daria Fleckenstein, Andreas Kaczmarczyk, Niklas Breitenbach-Netter, Gili Rosenberg, Roman Peter Jakob, Isabel Sorg, Amanzhol Kurmashev, Carlos Flores, Eva Jiménez-Siebert, Elinor Morris, Steffi Klimke, Sarah Tschudin-Sutter, Andreas Hierlemann, Timm Maier, Christoph Dehio, Urs Jenal, Knut Drescher

bioRxiv 2026.02.05.704027; doi: [https://doi.org/10.64898/2026.02.05.704027](https://doi.org/10.64898/2026.02.05.704027)

Version 0.1.0

# Instructions to reproduce figures

1. Install `snakemake` and `conda` on your system. See system requirements.
2. Clone this repository.
3. Copy the raw data from [https://doi.org/10.5281/zenodo.18486202](https://doi.org/10.5281/zenodo.18486202).
    1. Copy the contents of `Fig3.zip` to `data/raw/fig3`, do the same for `Fig4.zip` and `SuppFig8.zip`.
    2. Copy the contents of `Fig4_lbl.zip` to `data/preprocessed/fig4`, do the same for `SuppFig8_lbl.zip`.
    3. Copy the file `Fig.2_c_d.xlsx` into `data/preprocessed`.
4. Run the following commands to create figures
    ```shell
    snakemake --use-conda fig2 -j1  # for panels of figure 2
    ```

## System requirements

The software requires python and several python dependencies. These are organized via
`conda` and `snakemake`. Please install this software. `conda` can be installed via [miniforge](https://github.com/conda-forge/miniforge#install) (recommended) or [miniconda](https://continuumio-docs.readthedocs-hosted.com/miniconda/install/).
For `snakemake` follow
[https://snakemake.readthedocs.io/en/stable/getting_started/installation.html](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

## Project organization

```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── Snakefile
├── workflow           <- Snakemake workflow
│   ├── rules          <- Rules for Snakemake 
│   └── envs           <- Conda environment yaml files for Snakemake
├── data               <- All project data, ignored by git
│   ├── processed      <- Intermediate files
│   ├── preprocessed   <- Intermediate files that are externally supplied
│   │   ├── fig4       <- Label data for Fig. 4
│   │   └── suppfig8   <- Label data for Supp. Fig. 8
│   └── raw            <- The original, immutable data dump
│       ├── fig3       <- Raw files for Fig. 3
│       ├── fig4       <- Raw files for Fig. 4
│       └── suppfig8   <- Raw files for Supp. Fig. 8
├── results            <- Output files and figures
├── notebooks          <- Preliminary notebooks
└── src                <- Source code

```

### Code

Fig. 2c and Fig. 2d were calculated and plotted with `notebooks/2025-04-01-triton-macrophage-experiments.ipynb`.

Fig. 3f and Fig. 3g were computed and plotted with the snakemake workflow present under `workflow/rules/analysis.smk`. The python scripts under `src` belong to this workflow.

Fig. 4h was analyzed and plotted with `notebooks/2025-12-22-inverted-bladder-model.ipynb`. The supplementary figure including more results on the inverted bladder model was generated with the notebooks in `notebooks\inverted-bladder-model`.

### Data

The raw data is available under [https://doi.org/10.5281/zenodo.18486202](https://doi.org/10.5281/zenodo.18486202).

## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).
