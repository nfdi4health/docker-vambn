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


module modelling:
    snakefile:
        "modular-modelling.snakefile"
    config:
        config


use rule * from modelling as modular_modelling_*


################################################################################
# Modular Modelling
################################################################################


# Rule to generate all possible outputs
rule all:
    input:
        expand(
            "{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}{figure_prefix}",
            dataset_name=config["snakemake"]["DATASETS"],
            var=["wogan", "wgan"] if config["snakemake"]["with_gan"] else ["wogan"],
            mtl=["womtl", "wmtl"] if config["snakemake"]["with_mtl"] else ["womtl"],
            output_dir=config["snakemake"]["output_dir"],
            figure_prefix=[
                "/distributions.pdf",
                "_heatmap_virtual_all.pdf",
                # "_error.png",
                "_jsd_vir.png",
                "_jsd_dec.png",
                "_umap_decoded.png",
                "_tsne_decoded.png",
                "_jsd_by_module.png",
            ],
            shared=[
                "sharedLinear",
                "concatMtl",
                "concatIndiv",
                "none",
                "avgMtl",
                "maxMtl",
                "encoder",
                "encoderMtl"
            ],
        ),
        expand(
            "{output_dir}/optimization/modular_{shared}_{dataset_name}_{var}_{mtl}/study_plots",
            dataset_name=config["snakemake"]["DATASETS"],
            output_dir=config["snakemake"]["output_dir"],
            var=["wogan", "wgan"] if config["snakemake"]["with_gan"] else ["wogan"],
            mtl=["womtl", "wmtl"] if config["snakemake"]["with_mtl"] else ["womtl"],
            shared=[
                "sharedLinear",
                "concatMtl",
                "concatIndiv",
                "none",
                "avgMtl",
                "maxMtl",
                "encoder",
                "encoderMtl"
            ],
        ),
        expand(
            "{output_dir}/metrics/modular_{shared}_{dataset_name}_{var}_{mtl}/auc_metrics.csv",
            dataset_name=config["snakemake"]["DATASETS"],
            output_dir=config["snakemake"]["output_dir"],
            var=["wogan", "wgan"] if config["snakemake"]["with_gan"] else ["wogan"],
            mtl=["womtl", "wmtl"] if config["snakemake"]["with_mtl"] else ["womtl"],
            shared=[
                "sharedLinear",
                "concatMtl",
                "concatIndiv",
                "none",
                "avgMtl",
                "maxMtl",
            ],
        ),
        expand(
            "{output_dir}/syndat/modular_{shared}_{dataset_name}_{var}_{mtl}/raw_syndat.csv",
            dataset_name=config["snakemake"]["DATASETS"],
            output_dir=config["snakemake"]["output_dir"],
            var=["wogan", "wgan"] if config["snakemake"]["with_gan"] else ["wogan"],
            mtl=["womtl", "wmtl"] if config["snakemake"]["with_mtl"] else ["womtl"],
            shared=[
                "sharedLinear",
                "concatMtl",
                "concatIndiv",
                "none",
                "avgMtl",
                "maxMtl",
            ],
        )
    default_target: True


rule CompareDistributions:
    input:
        original_data=rules.modular_modelling_preprocessing_ConcatAndConvertRawFiles.output.concat,
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        decoded_data=rules.modular_modelling_GatherDecodedData.output.output_data,
        virtual_data=rules.modular_modelling_GenerateSyntheticData.output.output_data,
        script="vambn/visualization/distribution.py",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    output:
        complete="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}/distributions.pdf",
        decoded_only="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}/distributions_onlydecoded.pdf",
        metric_file="{output_dir}/metrics/modular_{shared}_{dataset_name}_{var}_{mtl}/jsd_metrics.json",
    shell:
        """
        python -m vambn.visualization.distribution {input.original_data} {input.decoded_data} {input.virtual_data} {output} {input.grouping} {wildcards.dataset_name} {wildcards.var}_{wildcards.mtl}
        """


rule CalculateMetrics:
    input:
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        original_data=rules.modular_modelling_preprocessing_ConcatAndConvertRawFiles.output.concat,
        decoded_data=rules.modular_modelling_GatherDecodedData.output.output_data,
        virtual_data=rules.modular_modelling_GenerateSyntheticData.output.output_data,
        script="vambn-r/make_corrplot.R",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    conda:
        config["snakemake"]["r_env"]
    output:
        all_virtual="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_heatmap_virtual_all.pdf",
        all_decoded="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_heatmap_decoded_all.pdf",
        cont_virtual="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_heatmap_virtual_cont.pdf",
        cont_decoded="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_heatmap_decoded_cont.pdf",
        results="{output_dir}/metrics/modular_{shared}_{dataset_name}_{var}_{mtl}/corr_metrics.json",
    log:
        "{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_heatmap.log",
    shell:
        """
        Rscript vambn-r/make_corrplot.R {input.grouping} {input.original_data} {input.decoded_data} {input.virtual_data} {output} {wildcards.dataset_name} {wildcards.var}_{wildcards.mtl} > {log} 2>&1
        """


rule GenerateErrorPlot:
    input:
        results=rules.modular_modelling_Train.output.results,
        script="vambn/visualization/generate_jsd_plot.py",
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
        loss_mode="[A-Za-z]+",
    threads: 1
    params:
        hivae="modular",
        experiment_name="modular_{dataset_name}_{var}_{mtl}",
    resources:
        runtime="10m",
    output:
        plot="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_error.png",
    log:
        "{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_error.log",
    shell:
        """
        python -m vambn.visualization.generate_jsd_plot {input.results} {output.plot} > {log} 2>&1
        """


rule JsdPlot_Vir:
    input:
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        original_data=rules.modular_modelling_preprocessing_ConcatAndConvertRawFiles.output.concat,
        virtual_data=rules.modular_modelling_GenerateSyntheticData.output.output_data,
        script="vambn/visualization/calculate_metrics.py",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    output:
        "{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_jsd_vir.png",
    log:
        "{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_jsd_vir.log",
    shell:
        """
        python -m vambn.visualization.calculate_metrics generate-jsd-plot {input.grouping} {input.original_data} {input.virtual_data} {output} > {log} 2>&1
        """


rule JsdPlot_Dec:
    input:
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        original_data=rules.modular_modelling_preprocessing_ConcatAndConvertRawFiles.output.concat,
        decoded_data=rules.modular_modelling_GatherDecodedData.output.output_data,
        script="vambn/visualization/calculate_metrics.py",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    output:
        "{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_jsd_dec.png",
    log:
        "{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_jsd_dec.log",
    shell:
        """
        python -m vambn.visualization.calculate_metrics generate-jsd-plot {input.grouping} {input.original_data} {input.decoded_data} {output} > {log} 2>&1
        """


rule GenerateGraph:
    input:
        bootstrap_file=rules.modular_modelling_GenerateBayesianNetwork.output.bootstrap_out,
        script="vambn-r/vis_graph.R",
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    params:
        r_module=config["snakemake"]["cluster_modules"]["R"],
        r_env=config["snakemake"]["r_env"],
    threads: 1
    conda:
        config["snakemake"]["r_env"]
    output:
        "{output_dir}/bn/modular_{shared}_{dataset_name}_{var}_{mtl}/clean_graph.svg",
    shell:
        """
        if [ {params.r_module} != "None" ]; then
            module load {params.r_module}
        fi

        Rscript {input.script} {input.bootstrap_file} {output}
        """


rule GenerateOptunaPlots:
    input:
        proxy=rules.modular_modelling_Optimize.output.parameter_file,
        script="vambn/visualization/generate_optuna_plots.py",
    threads: 1
    params:
        experiment_name=lambda x: f"modular_{x.shared}_{x.dataset_name}_{x.var}_{x.mtl}_{config['general']['logging']['mlflow']['experiment_name']}",
        database="sqlite:///{output_dir}/optimization/modular_{shared}_{dataset_name}_{var}_{mtl}/optuna.db"
        if config["general"]["optuna_db"] is None
        else config["general"]["optuna_db"],
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    output:
        folder=directory(
            "{output_dir}/optimization/modular_{shared}_{dataset_name}_{var}_{mtl}/study_plots"
        ),
    shell:
        """
        python -m vambn.visualization.generate_optuna_plots {params.database} {params.experiment_name} {output.folder}
        """

rule CalculateAuc:
    input:
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        original_data=rules.modular_modelling_preprocessing_ConcatAndConvertRawFiles.output.concat,
        decoded_data=rules.modular_modelling_GatherDecodedData.output.output_data,
        virtual_data=rules.modular_modelling_GenerateSyntheticData.output.output_data,
        script="vambn/visualization/calculate_metrics.py",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z]*_VIS[0-9]+",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    output:
        metric_file="{output_dir}/metrics/modular_{shared}_{dataset_name}_{var}_{mtl}/auc_metrics.csv",
    shell:
        """
        python -m vambn.visualization.calculate_metrics calculate-auc {input.grouping} {input.original_data} {input.decoded_data} {input.virtual_data} {output}
        """


rule GenerateUmap:
    input:
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        original_data=rules.modular_modelling_preprocessing_ConcatAndConvertRawFiles.output.concat,
        decoded_data=rules.modular_modelling_GatherDecodedData.output.output_data,
        virtual_data=rules.modular_modelling_GenerateSyntheticData.output.output_data,
        script="vambn/visualization/generate_umap_plot.py",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z]*_VIS[0-9]+",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    output:
        decoded_file="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_umap_decoded.png",
        virtual_file="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_umap_virtual.png",
    shell:
        """
        python -m vambn.visualization.generate_umap_plot {input.grouping} {input.original_data} {input.decoded_data} {input.virtual_data} {output}
        """

rule GenerateTsne:
    input:
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        original_data=rules.modular_modelling_preprocessing_ConcatAndConvertRawFiles.output.concat,
        decoded_data=rules.modular_modelling_GatherDecodedData.output.output_data,
        virtual_data=rules.modular_modelling_GenerateSyntheticData.output.output_data,
        script="vambn/visualization/generate_tsne_plot.py",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z]*_VIS[0-9]+",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    output:
        decoded_file="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_tsne_decoded.png",
        virtual_file="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_tsne_virtual.png",
    shell:
        """
        python -m vambn.visualization.generate_tsne_plot {input.grouping} {input.original_data} {input.decoded_data} {input.virtual_data} {output}
        """


rule GenerateJsdByModule:
    input:
        grouping=rules.modular_modelling_preprocessing_Preprocessing.input.grouping,
        jsd_metrics=rules.CompareDistributions.output.metric_file,
        script="vambn/visualization/jsd_by_module.py",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
        var="[A-Za-z]*",
        mtl="[A-Za-z]*",
        output_dir="[A-Za-z0-9]*",
        shared="[A-Za-z]*",
    output:
        output_plot="{output_dir}/figures/modular_{shared}_{dataset_name}_{var}_{mtl}_jsd_by_module.png",
    shell:
        """
        python -m vambn.visualization.jsd_by_module {input.jsd_metrics} {input.grouping} {output.output_plot}
        """
        