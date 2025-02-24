from glob import glob
import re
from typing import Set
import random
from itertools import combinations
from snakemake.utils import min_version


min_version("6.0")
random.seed(42)


configfile: "vambn_config.yml"


################################################################################
# Module dependencies
################################################################################


if config["snakemake"]["modules"]["use_modular"]:

    module modular:
        snakefile:
            "snakemake_modules/modular-postprocessing.snakefile"
        config:
            config

    use rule * from modular as modular_*


if config["snakemake"]["modules"]["use_traditional"]:

    module traditional:
        snakefile:
            "snakemake_modules/traditional-postprocessing.snakefile"
        config:
            config

    use rule * from traditional as traditional_*


if (
    config["snakemake"]["modules"]["use_traditional"]
    and config["snakemake"]["modules"]["use_modular"]
):

    ruleorder: modular_modular_modelling_preprocessing_setupEnv_setupR > traditional_traditional_modelling_preprocessing_setupEnv_setupR > modular_modular_modelling_preprocessing_Preprocessing > traditional_traditional_modelling_preprocessing_Preprocessing > modular_modular_modelling_preprocessing_ConcatImputedFiles > traditional_traditional_modelling_preprocessing_ConcatImputedFiles > modular_modular_modelling_preprocessing_ConcatStaloneFiles > traditional_traditional_modelling_preprocessing_ConcatStaloneFiles > modular_modular_modelling_preprocessing_ConcatRawFiles > traditional_traditional_modelling_preprocessing_ConcatRawFiles > modular_modular_modelling_preprocessing_setupEnv_initializeMlflowExperiment > traditional_traditional_modelling_preprocessing_setupEnv_initializeMlflowExperiment


################################################################################
# Modular Modelling
################################################################################


# Rule to generate all possible outputs
# Collect targets from active modules
module_inputs = []
if config["snakemake"]["modules"]["use_modular"]:
    module_inputs.extend(rules.modular_all.input)
if config["snakemake"]["modules"]["use_traditional"]:
    module_inputs.extend(rules.traditional_all.input)


rule all:
    input:
        module_inputs,
        expand(
            "{output_dir}/metrics/{dataset}_aggregated_metrics.csv",
            dataset=config["snakemake"]["DATASETS"],
            output_dir=config["snakemake"]["output_dir"],
        ),
    default_target: True


MODULES = []

MODULES = []
if config["snakemake"]["modules"]["use_modular"]:
    MODULES.extend(
        [
            "modular_concatMtl",
            "modular_concatIndiv",
            "modular_none",
            "modular_avgMtl",
            "modular_maxMtl",
            "modular_sharedLinear",
            "modular_encoderMtl",
            "modular_encoder",
        ]
    )
if config["snakemake"]["modules"]["use_traditional"]:
    MODULES.extend(
        [
            "traditional",
        ]
    )


VARS = ["wogan", "wgan"] if config["snakemake"]["with_gan"] else ["wogan"]
MTLS = ["womtl", "wmtl"] if config["snakemake"]["with_mtl"] else ["womtl"]


def get_results_files(wildcards):
    return [
        f"{wildcards.output_dir}/metrics/{module}_{wildcards.dataset}_{var}_{mtl}/{metric_file}"
        for module in MODULES
        for var in VARS
        for mtl in MTLS
        for metric_file in ["auc_metrics.csv", "corr_metrics.json", "jsd_metrics.json"]
    ]


rule aggregate_data:
    input:
        files=get_results_files,
        grouping="data/raw/grouping_{dataset}.csv",
        file="vambn/visualization/generate_results_plot.py",
    output:
        "{output_dir}/metrics/{dataset}_aggregated_metrics.csv",
        "{output_dir}/figures/{dataset}_metrics.png",
    shell:
        "python -m vambn.visualization.generate_results_plot {input.grouping} {input.files} {output}"
