# Monitoring intracellular antibiotic concentrations in real-time using allosteric biosensors

Code repository for

Monitoring intracellular antibiotic concentrations in real-time using allosteric biosensors

Daria Fleckenstein, Andreas Kaczmarczyk, Niklas Breitenbach-Netter, Gili Rosenberg, Roman Peter Jakob, Isabel Sorg, Amanzhol Kurmashev, Carlos Flores, Eva Jiménez-Siebert, Elinor Morris, Steffi Klimke, Sarah Tschudin-Sutter, Andreas Hierlemann, Timm Maier, Christoph Dehio, Urs Jenal, Knut Drescher

bioRxiv 2026.02.05.704027; doi: [https://doi.org/10.64898/2026.02.05.704027](https://doi.org/10.64898/2026.02.05.704027)

Version 0.1.0

## Project organization

```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── Snakefile
├── workflow           <- Snakemake workflow
│   ├── rules          <- Additional rules for Snakemake (HW)
│   └── envs           <- Conda environment yaml files for Snakemake (HW)
├── data               <- All project data, ignored by git
│   ├── processed      <- The final, canonical data sets for modeling. (PG)
│   └── raw            <- The original, immutable data dump. (RO)
├── notebooks          <- Jupyter notebooks for prototyping
└── src                <- Source code for this project (HW)

```

Please use symbolic links to point `data/raw` towards the downloaded nd2 files from Fig3.zip or create the folder `data/raw` and copy the nd2 files from Fig3.zip into it.

### Data repository

The raw available under [https://doi.org/10.5281/zenodo.18486202](https://doi.org/10.5281/zenodo.18486202).

## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).
