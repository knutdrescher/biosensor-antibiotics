def nd2_input(wildcards):
    if wildcards.file == "NCCRBM128_sample4":
        return [
            "data/raw/suppfig8/NCCRBM128_sample4_before.nd2",
            "data/raw/suppfig8/NCCRBM128_sample4_TMP.nd2",
        ]
    else:
        return f"data/raw/suppfig8/{wildcards.file}.nd2"
rule align_translation:
    input:
        nd2=nd2_input,
        helper_script="src/suppfig8_helpers.py",
    output:
        npy="data/processed/suppfig8/{file}_pos{pos}.npy",
    conda: "../envs/suppfig8.yaml"
    script: "../../src/align_translation.py"

rule apply_translation_lbl:
    input:
        lbls=expand(
            "data/preprocessed/suppfig8/{{file}}_pos{{pos}}_t{t}_mScarlett.tiff",
            t=range(2),
        ),
        npy=rules.align_translation.output.npy,
        helper_script="src/suppfig8_helpers.py",
    output:
        tif="data/processed/suppfig8/{file}_pos{pos}_lbl.tif",
    conda: "../envs/suppfig8.yaml"
    script: "../../src/apply_translation_lbl.py"

rule suppfig8_analyse:
    input:
        nd2=nd2_input,
        lbls=expand(
            "data/preprocessed/suppfig8/{{file}}_pos{{pos}}_t{t}_mScarlett.tiff",
            t=range(2),
        ),
        aligned_lbls=rules.apply_translation_lbl.output.tif,
        helper_script="src/suppfig8_helpers.py",
    output:
        csv="data/processed/suppfig8/{file}_pos{pos}.csv",
    conda: "../envs/suppfig8.yaml"
    script: "../../src/suppfig8_analyse.py"

def input_csvs(wildcards):
    files = []
    for sample in [4, 5, 6]:
        file = f"NCCRBM128_sample{sample}"
        for pos in range(4):
            if sample in [5, 6] and pos == 0:
                continue
            files.append(f"data/processed/suppfig8/{file}_pos{pos}.csv")
    return files
rule suppfig8:
    input:
        csvs=input_csvs,
    output:
        pdf="results/suppfig8/panel_b.pdf",
        txt="results/suppfig8/panel_b_stats.txt",
        xlsx="results/suppfig8/panel_b_source.xlsx",
    conda: "../envs/suppfig8.yaml"
    script: "../../src/suppfig8.py"
