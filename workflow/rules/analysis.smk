import os
from glob import glob
from pathlib import Path


rule segmentation:
    input:
        nd2="data/raw/{file}.nd2",
        script="src/segmentation.py",
    output:
        lbl="data/raw/{file}_lbl.tif",
    conda: "../envs/image-analysis.yaml"
    script: "../../src/segmentation.py"

def lbl_input_func(wildcards):
    file = wildcards.file
    auto = f"{file}_lbl.tif"
    manual = f"{file}_manual-lbl.tif"
    path = "data/raw"
    auto = os.path.join(path, auto)
    manual = os.path.join(path, manual)
    if Path(manual).exists():
        return manual
    else:
        return auto
rule edt_analysis_heatmap_calculation:
    input:
        # lbl=rules.segmentation.output.lbl,
        lbl=lbl_input_func,
        nd2=rules.segmentation.input.nd2,
        # script="src/edt_analysis.py",
        helpers="src/helpers.py",
    output:
        i_mean="data/processed/nd2/{file}/i_mean.npy",
        i_std="data/processed/nd2/{file}/i_std.npy",
        deriv_mean="data/processed/nd2/{file}/deriv_mean.npy",
        deriv_std="data/processed/nd2/{file}/deriv_std.npy",
        exp_times="data/processed/nd2/{file}/exp_times.npy",
        deriv_time="data/processed/nd2/{file}/deriv_time.npy",
        x="data/processed/nd2/{file}/x.npy",
        edt="data/raw/{file}_edt.tif",
    conda: "../envs/image-analysis.yaml"
    script: "../../src/edt_analysis.py"

rule plot_qc_heatmap:
    input:
        i_mean=rules.edt_analysis_heatmap_calculation.output.i_mean,
        i_std=rules.edt_analysis_heatmap_calculation.output.i_std,
        deriv_mean=rules.edt_analysis_heatmap_calculation.output.deriv_mean,
        deriv_std=rules.edt_analysis_heatmap_calculation.output.deriv_std,
        exp_times=rules.edt_analysis_heatmap_calculation.output.exp_times,
        deriv_time=rules.edt_analysis_heatmap_calculation.output.deriv_time,
        x=rules.edt_analysis_heatmap_calculation.output.x,
        script="src/plot_qc_heatmap.py",
    output:
        png="data/raw/{file}_qc.png",
    conda: "../envs/plotting.yaml"
    script: "../../src/plot_qc_heatmap.py"

rule slope:
    input:
        heatmap=rules.edt_analysis_heatmap_calculation.output.deriv_mean,
        time=rules.edt_analysis_heatmap_calculation.output.deriv_time,
        x=rules.edt_analysis_heatmap_calculation.output.x,
        script="src/slope.py",
    output:
        yaml="data/processed/nd2/{file}/slope.yaml",
        png="data/raw/{file}_slope.png",
    conda: "../envs/plotting.yaml"
    script: "../../src/slope.py"

rule penetration_time:
    input:
        heatmap=rules.edt_analysis_heatmap_calculation.output.deriv_mean,
        time=rules.edt_analysis_heatmap_calculation.output.deriv_time,
        x=rules.edt_analysis_heatmap_calculation.output.x,
        yaml="data/raw/{file}_penetration_time.yaml",
        script="src/penetration_time.py",
    output:
        yaml="data/processed/nd2/{file}/penetration_time.yaml",
        png="data/raw/{file}_penetration_time.png",
    conda: "../envs/plotting.yaml"
    script: "../../src/penetration_time.py"

def passes(f):
    name = Path(f).stem
    if "control" in name:
        return False
    elif name in [
        "TMP-FAST_600_4", "TMP-FAST_500_1"
    ]:
        return False
    elif "HMBR" in name:
        return False
    return True

def gather_slopes_input(wildcards):
    files = glob("data/raw/*nd2")
    path = "data/processed/nd2"
    return [os.path.join(path, Path(f).stem + "/slope.yaml") for f in files if passes(f)]
rule gather_slopes:
    input:
        yamls=gather_slopes_input,
        HMBR=expand(
            "data/processed/nd2/{file}/slope.yaml",
            file=["TMP-FAST_400_control_HMBR_3", "TMP-FAST_300_control_HMBR_1"],
        ),
        script="src/plot_slopes.py"
    output:
        csv="data/processed/slopes.csv",
        png="data/raw/slopes.png",
    conda: "../envs/plotting.yaml"
    script: "../../src/plot_slopes.py"

def gather_penetration_input(wildcards):
    files = glob("data/raw/*nd2")
    path = "data/processed/nd2"
    return [os.path.join(path, Path(f).stem + "/penetration_time.yaml") for f in files if passes(f)]
rule gather_penetration:
    input:
        yamls=gather_penetration_input,
        script="src/plot_penetration_times.py",
        HMBR=expand(
            "data/processed/nd2/{file}/penetration_time.yaml",
            file=["TMP-FAST_400_control_HMBR_3", "TMP-FAST_300_control_HMBR_1"],
        ),
    output:
        csv="data/processed/penetration_times.csv",
        png="data/raw/penetration_times.png",
    conda: "../envs/plotting.yaml"
    script: "../../src/plot_penetration_times.py"

def gather_input(wildcards):
    files = glob("data/raw/*nd2")
    return [f.replace(".nd2", "_slope.png") for f in files] + [
        f.replace(".nd2", "_penetration_time.png") for f in files
    ]
rule gather:
    input:
        files=gather_input,
        gather_slopes=rules.gather_slopes.output.png,
        gather_penetration=rules.gather_penetration.output.png,

files = [
    "0TMP-FAST_50um_1",
    "0TMP-FAST_80um_2",
    "0TMP-FAST_200um_3",
    "0TMP-FAST_400um_3",
]
additional_files = [
    "pFAST_control",
    "TMP-FAST_300_control_HMBR_1",
]
rule figure:
    output:
        png="data/raw/figure/figure.png",
        svg="data/raw/figure/figure.svg",
        pdf="data/raw/figure/figure.pdf",
    input:
        nd2_gfp=expand("data/raw/{file}.nd2", file=files),
        nd2_red=expand("data/raw/mScarlet/{file}.nd2", file=
            [
                "TMP-FAST_50_1_mScarlet",
                "TMP-FAST_80_2_mScarlet",
                "TMP-FAST_200_3_mScarlet",
                "TMP-FAST_400_3_mScarlet",
            ],
        ),
        edt=expand("data/raw/{file}_edt.tif", file=files),
        heatmaps=expand("data/processed/nd2/{file}/i_mean.npy",
                        file=files+additional_files),
        derivs=expand("data/processed/nd2/{file}/deriv_mean.npy",
                      file=files+additional_files),
        deriv_ts=expand("data/processed/nd2/{file}/deriv_time.npy",
                        file=files+additional_files),
        ts=expand("data/processed/nd2/{file}/exp_times.npy", file=files+additional_files),
        xs=expand("data/processed/nd2/{file}/x.npy", file=files+additional_files),
        slopes=rules.gather_slopes.output.csv,
        times=rules.gather_penetration.output.csv,
        script="src/figure.py",
    conda: "../envs/plotting.yaml"
    script: "../../src/figure.py"

rule HMBR:
    input:
        a="data/raw/TMP-FAST_400_control_HMBR_3_penetration_time.png",
        b="data/raw/TMP-FAST_300_control_HMBR_1_penetration_time.png",
        c="data/raw/TMP-FAST_400_control_HMBR_3_slope.png",
        d="data/raw/TMP-FAST_300_control_HMBR_1_slope.png",
