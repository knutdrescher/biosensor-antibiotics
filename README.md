# Monitoring intracellular antibiotic concentrations in real-time using allosteric biosensors

Code repository for

Monitoring intracellular antibiotic concentrations in real-time using allosteric biosensors

Daria Fleckenstein, Andreas Kaczmarczyk, Niklas Breitenbach-Netter, Gili Rosenberg, Roman Peter Jakob, Isabel Sorg, Amanzhol Kurmashev, Carlos Flores, Eva Jiménez-Siebert, Elinor Morris, Steffi Klimke, Sarah Tschudin-Sutter, Andreas Hierlemann, Timm Maier, Christoph Dehio, Urs Jenal, Knut Drescher

bioRxiv 2026.02.05.704027; doi: [https://doi.org/10.64898/2026.02.05.704027](https://doi.org/10.64898/2026.02.05.704027)

Version 0.1.0

# Instructions to reproduce figures

The software in this repository produces the panels Fig. 2c, Fig. 2d, Fig. 3f, Fig. 3g and Fig. 4h as well as Supp. Fig. 8b.

1. Install `snakemake` and `conda` on your system. See system requirements. This should take ~20 minutes.
2. Clone this repository.
3. Copy the raw data from [https://doi.org/10.5281/zenodo.18486202](https://doi.org/10.5281/zenodo.18486202).
    1. Copy the contents of `Fig3.zip` to `data/raw/fig3`, do the same for `Fig4.zip` and `SuppFig8.zip`.
    2. Copy the contents of `Fig4_lbl.zip` to `data/preprocessed/fig4`, do the same for `SuppFig8_lbl.zip`.
    3. Copy the file `Fig.2_c_d.xlsx` into `data/preprocessed`.
4. Run the following commands to create figures
    ```shell
    # for panels of figure 2
    # less than 5 minutes, including automatic installation of additional packages
    snakemake --use-conda fig2 -j1  

    # for panel h of figure 4
    # ~20 minutes, including automatic installation of additional packages
    snakemake --use-conda fig4 -j3  
    ```
5. Have a look at the outputs in `results` folder.

## System requirements

The software requires python and several python dependencies (see `workflow/envs` folder). These are organized via
`conda` and `snakemake`. Please install this software. `conda` can be installed via [miniforge](https://github.com/conda-forge/miniforge#install) (recommended) or [miniconda](https://continuumio-docs.readthedocs-hosted.com/miniconda/install/).
For `snakemake` follow
[https://snakemake.readthedocs.io/en/stable/getting_started/installation.html](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

The software has been tested with `conda==25.5.0` and `snakemake==9.9.0` on `Ubuntu 22.04.1 LTS` on WSL2 and with `conda==` and `snakemake==` on Windows 11.

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

Fig. 3f and Fig. 3g were computed and plotted with the snakemake workflow present under `workflow/rules/analysis.smk`. The python scripts under `src` belong to this workflow.

Fig. 4h was analyzed and plotted with `notebooks/2025-12-22-inverted-bladder-model.ipynb`. The supplementary figure including more results on the inverted bladder model was generated with the notebooks in `notebooks\inverted-bladder-model`.

### Data

The raw data is available under [https://doi.org/10.5281/zenodo.18486202](https://doi.org/10.5281/zenodo.18486202).

## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).
