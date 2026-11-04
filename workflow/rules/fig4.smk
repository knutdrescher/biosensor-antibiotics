rule preprocess_fig4:
    input:
        nd2="data/raw/fig4/{file}.nd2",
        lbls=expand(
            "data/preprocessed/fig4/{{file}}_t{t}_m{m}_mScarlet.tiff",
            m=range(3),
            t=range(2),
        ),
    output:
        csv="data/processed/fig4/{file}.csv",
    conda: "../envs/fig4.yaml"
    script: "../../src/preprocess_fig4.py"

rule fig4:
    input:
        csvs=expand(
            "data/processed/fig4/Fig.4_h_TrimFABS_ANOX_mScarlet_insert{i}.csv",
            i=[1, 3, 4],
        ),
    output:
        svg_all="results/fig4/violin_plots_separated.svg",
        svg="results/fig4/violin_plots.svg",
        pdf="results/fig4/panel_h.pdf",
        xlsx_all="results/fig4/all_medians.xlsx",
        xlsx="results/fig4/medians_source_file_fig4h.xlsx",
    conda: "../envs/fig4.yaml"
    script: "../../src/fig4.py"
