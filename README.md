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
├── requirements.txt
├── Snakefile
├── workflow           <- Snakemake workflow
│   ├── rules          <- Additional rules for Snakemake (HW)
│   └── envs           <- Conda environment yaml files for Snakemake (HW)
├── bin                <- Compiled and external code, ignored by git (PG)
│   └── external       <- Any external source code, ignored by git (RO)
├── config             <- Configuration files (HW)
├── data               <- All project data, ignored by git
│   ├── processed      <- The final, canonical data sets for modeling. (PG)
│   ├── raw            <- The original, immutable data dump. (RO)
│   └── temp           <- Intermediate data that has been transformed. (PG)
├── docs               <- Documentation notebook for users (HW)
│   ├── manuscript     <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports        <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── notebooks          <- Jupyter notebooks for prototyping
├── results
│   ├── figures        <- Figures for the manuscript or reports (PG)
│   └── output         <- Other output for the manuscript or reports (PG)
├── src                <- Source code for this project (HW)
└── logs               <- Log files (from Snakemake) (PG)

```


## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).
