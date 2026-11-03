rule fig2:
    input:
        xlsx="data/preprocessed/Fig.2_c_d.xlsx",
    output:
        panel_c="results/fig2/panel_c.pdf",
        panel_d="results/fig2/panel_d.pdf",
        xlsx="results/fig2/triton-times.xlsx",
        pvalue="results/fig2/pvalues.txt",
    conda: "../envs/fig2.yaml"
    script: "../../src/fig2.py"
