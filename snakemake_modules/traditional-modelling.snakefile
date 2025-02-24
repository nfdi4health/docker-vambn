from glob import glob
import re
from typing import Set
import random
from itertools import combinations
from snakemake.utils import min_version
import sys
import pandas as pd

args = sys.argv
if "--configfile" in args:
    config_path = args[args.index("--configfile") + 1]
elif "--configfiles" in args:
    config_path = args[args.index("--configfiles") + 1]
else:
    config_path = "vambn_config.yml"

min_version("6.0")
random.seed(42)


configfile: "vambn_config.yml"


################################################################################
# Module dependencies
################################################################################


module preprocessing:
    snakefile:
        "preprocessing.snakefile"
    config:
        config


use rule * from preprocessing as preprocessing_*


################################################################################
# Traditional Modelling
################################################################################


# Rule to generate all possible outputs
rule all:
    input:
        expand(
            "{output_dir}/fit/traditional_{dataset_name}_{var}_{mtl}/overall_metrics.csv",
            dataset_name=config["snakemake"]["DATASETS"],
            output_dir=config["snakemake"]["output_dir"],
            var=["wogan", "wgan"] if config["snakemake"]["with_gan"] else ["wogan"],
            mtl=["womtl", "wmtl"] if config["snakemake"]["with_mtl"] else ["womtl"],
        ),
        expand(
            "{output_dir}/syndat/traditional_{dataset_name}_{var}_{mtl}/synthetic_syndat.csv",
            dataset_name=config["snakemake"]["DATASETS"],
            output_dir=config["snakemake"]["output_dir"],
            var=["wogan", "wgan"] if config["snakemake"]["with_gan"] else ["wogan"],
            mtl=["womtl", "wmtl"] if config["snakemake"]["with_mtl"] else ["womtl"],
        ),
    default_target: True


def get_number_of_cols(wildcards):
    grouping = pd.read_csv(f"data/raw/grouping_{wildcards.dataset_name}.csv")
    n_feat = (grouping["technical_group_name"] == wildcards.module).sum()
    return n_feat


rule Optimize:
    input:
        data=rules.preprocessing_Preprocessing.output.folder,
        config=config_path,
        script="vambn/modelling/run_model.py",
    params:
        experiment_name=lambda x: f"traditional_{x.dataset_name}_{x.module}_{x.var}_{x.mtl}_{config['general']['logging']['mlflow']['experiment_name']}",
        n_trials=config["optimization"]["n_traditional_trials"],
        checkpoint_path="{output_dir}/optimization/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/checkpoints",
        database=(
            "{output_dir}/optimization/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/optuna.db"
            if config["general"]["optuna_db"] is None
            else config["general"]["optuna_db"]
        ),
        cli_arg=lambda x: "gan_optimize" if x.var == "wgan" else "loptimize",
        python_command="srun python" if config["snakemake"]["use_slurm"] else "python",
    threads: 6
    log:
        base="logs/{output_dir}/opt_{model_variant,traditional}_{dataset_name}_{var}_{mtl}_{module}.txt",
        stdout="logs/{output_dir}/opt_{model_variant,traditional}_{dataset_name}_{var}_{mtl}_{module}.stdout",
    resources:
        runtime="48h",
        mem_mb=8000,
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
        module="[A-Za-z_0-9]+",
    output:
        parameter_file="{output_dir}/optimization/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/optimization_results.json",
    shell:
        """
        performed_trials=$(python -m vambn.utils.trial_counter {params.database} {params.experiment_name})
        trials_to_do=$(( {params.n_trials} - $performed_trials))
        echo "Performed $performed_trials"
        echo "Run $trials_to_do further trials"
        {params.python_command} -m vambn.modelling.run_model hivae {params.cli_arg} {wildcards.module} {input.config} {input.data} {params.experiment_name} \
        {threads} {params.checkpoint_path} {params.database} \
        {output.parameter_file} --num-trials=$trials_to_do --log-file={log.base} > {log.stdout} 2>&1
        """


def get_opt_files(wildcards):
    file_name = rules.preprocessing_Preprocessing.input.groups.format(**vars(wildcards))
    with open(file_name, "r") as f:
        groups = f.read().splitlines()

    modules = [f"{x}" for x in groups for i in range(1)]

    return [
        f"{wildcards.output_dir}/optimization/{model_variant, traditional}_{wildcards.dataset_name}/{module}/optimization_results.json"
        for module in modules
    ]


rule GetOptFiles:
    input:
        get_opt_files,
    output:
        opt_files="{output_dir}/optimization/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/.performed_optimization",


rule Train:
    input:
        data=rules.preprocessing_Preprocessing.output.folder,
        config=config_path,
        script="vambn/modelling/run_model.py",
        parameter_file=rules.Optimize.output.parameter_file,
    params:
        checkpoint_path="{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/checkpoints",
        cli_arg=lambda x: "gan_train" if x.var == "wgan" else "ltrain",
        python_command="srun python" if config["snakemake"]["use_slurm"] else "python",
    threads: 6
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
        module="[A-Za-z_0-9]+",
    resources:
        tmpdir=lambda wildcards: f"/tmp/traditional_{wildcards.dataset_name}",
        mem_mb=8000,
        runtime="8h",
    log:
        base="logs/{output_dir}/fit_{model_variant,traditional}_{dataset_name}_{var}_{mtl}_{module}.txt",
        stdout="logs/{output_dir}/fit_{model_variant,traditional}_{dataset_name}_{var}_{mtl}_{module}.stdout",
    output:
        results="{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/overall_metrics.csv",
        metaenc="{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/data_outputs/meta_enc.csv",
        decoded_folder=directory(
            "{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}"
        ),
        data_outputs=directory(
            "{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/data_outputs"
        ),
        model_file="{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/model.bin",
        trainer_file="{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}/trainer.pkl",
    shell:
        """
        {params.python_command} -m vambn.modelling.run_model hivae {params.cli_arg} {wildcards.module} {input.config} {input.data}    {threads} \
        {params.checkpoint_path} {input.parameter_file} {output.decoded_folder} --log-file={log.base}  > {log.stdout} 2>&1
        """


def get_metaenc_files(wildcards):
    file_name = rules.preprocessing_Preprocessing.input.groups.format(**vars(wildcards))
    with open(file_name, "r") as f:
        groups = f.read().splitlines()

    modules = [f"{x}" for x in groups for i in range(1)]

    return [
        f"{wildcards.output_dir}/fit/traditional_{wildcards.dataset_name}_{wildcards.var}_{wildcards.mtl}/{module}/data_outputs"
        for module in modules
    ]


rule GatherDecodedData:
    input:
        decoded_folder=get_metaenc_files,
        stalone_data=rules.preprocessing_ConcatStaloneFiles.output.concat,
        script="vambn/data/make_data.py",
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
        module="[a-zA-Z0-9]*",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
    threads: 1
    output:
        output_data="{output_dir}/decoded/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/data.csv",
    shell:
        """
        python -m vambn.data.make_data gather traditional {input.decoded_folder} {input.stalone_data} {output.output_data}
        """


def get_result_files(wildcards):
    file_name = rules.preprocessing_Preprocessing.input.groups.format(**vars(wildcards))
    with open(file_name, "r") as f:
        groups = f.read().splitlines()

    modules = [f"{x}" for x in groups for i in range(1)]

    return [
        f"{wildcards.output_dir}/fit/{model_variant, traditional}_{wildcards.dataset_name}_{wildcards.var}_{wildcards.mtl}/{module}/overall_metrics.csv"
        for module in modules
    ]


rule MergeResults:
    input:
        files=get_result_files,
    output:
        results="{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/overall_metrics.csv",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z0-9]*",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    run:
        import pandas as pd
        from functools import reduce

        df = pd.concat([pd.read_csv(f) for f in input["files"]])

        df.to_csv(output["results"], index=False)


def get_encoding_files(wildcards):
    file_name = rules.preprocessing_Preprocessing.input.groups.format(**vars(wildcards))
    with open(file_name, "r") as f:
        groups = f.read().splitlines()

    modules = [f"{x}" for x in groups for i in range(1)]

    return [
        f"{wildcards.output_dir}/fit/traditional_{wildcards.dataset_name}_{wildcards.var}_{wildcards.mtl}/{module}/data_outputs/meta_enc.csv"
        for module in modules
    ]


rule MergeEncodings:
    input:
        files=get_encoding_files,
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
        module="[a-zA-Z0-9]*",
    output:
        metaenc="{output_dir}/fit/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/meta_enc.csv",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    run:
        import pandas as pd

        results = [
            pd.read_csv(file) if i == 0 else pd.read_csv(file).drop("SUBJID", axis=1)
            for i, file in enumerate(input["files"])
        ]

        pd.concat(results, axis=1).to_csv(output["metaenc"], index=False)


rule GenerateBayesianNetwork:
    input:
        stalone_data=rules.preprocessing_ConcatStaloneFiles.output.concat,
        metaenc=rules.MergeEncodings.output.metaenc,
        grouping=rules.preprocessing_Preprocessing.input.grouping,
        blacklist=rules.preprocessing_Preprocessing.input.blacklist,
        whitelist=rules.preprocessing_Preprocessing.input.whitelist,
        start_dag=rules.preprocessing_Preprocessing.input.start_dag,
        script="vambn-r/bayesian_net.R",
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
        module="[a-zA-Z0-9]*",
    params:
        refactor="--refactor" if config["snakemake"]["bn"]["refactor"] else "",
        bnl_cv_runs=config["snakemake"]["bn"]["cv_runs"],
        bnl_cv_restart=config["snakemake"]["bn"]["cv_restart"],
        bnl_fit=config["snakemake"]["bn"]["fit"],
        bnl_maxp=config["snakemake"]["bn"]["maxp"],
        bnl_loss=(
            "--bnl_loss=" + config["snakemake"]["bn"]["loss"]
            if config["snakemake"]["bn"]["loss"]
            else ""
        ),
        bnl_score=config["snakemake"]["bn"]["score"],
        n_bootstrap=config["snakemake"]["bn"]["n_bootstrap"],
        seed=config["snakemake"]["bn"]["seed"],
        r_module=config["snakemake"]["cluster_modules"]["R"],
        r_env=config["snakemake"]["r_env"],
    output:
        output_dir=directory(
            "{output_dir}/bn/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/"
        ),
        bn_out="{output_dir}/bn/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/bn.rds",
        bootstrap_out="{output_dir}/bn/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/bootstrap_strength.csv",
        likelihood_out="{output_dir}/bn/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/likelihood_real.csv",
        bn_data="{output_dir}/bn/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/bn_data.rds",
    threads: 8
    resources:
        time="12:00:00",
        mem_mb=8000,
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    conda:
        config["snakemake"]["r_env"]
    shell:
        """
        if [ {params.r_module} != "None" ]; then
            module load {params.r_module}
        fi

        Rscript {input.script} \
        --stalone_data={input.stalone_data} \
        --grouping_file={input.grouping} \
        --metaenc={input.metaenc} \
        --blacklist={input.blacklist} \
        --whitelist={input.whitelist} \
        --start_dag={input.start_dag} \
        --bnl_maxp={params.bnl_maxp} \
        --n_bootstrap={params.n_bootstrap} \
        --seed={params.seed} \
        --output_dir={output.output_dir} \
        --bn_out={output.bn_out} \
        --likelihood_out={output.likelihood_out} \
        --bn_data={output.bn_data} \
        --cores={threads} 
        """


rule GenerateSyntheticPatients:
    input:
        fitted=rules.GenerateBayesianNetwork.output.bn_out,
        script="vambn-r/generate_syn_patients.R",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z0-9]*",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    params:
        n_patients=1000,
        r_module=config["snakemake"]["cluster_modules"]["R"],
        r_env=config["snakemake"]["r_env"],
    conda:
        config["snakemake"]["r_env"]
    output:
        encodings="{output_dir}/synthetic/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/synthetic_meta_enc.csv",
        likelihood_synthetic="{output_dir}/bn/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/likelihood_synthetic.csv",
    shell:
        """
        if [ {params.r_module} != "None" ]; then
            module load {params.r_module}
        fi

        Rscript {input.script} --fitted_model={input.fitted} --n_patients={params.n_patients} --encoding={output.encodings} --likelihood_out={output.likelihood_synthetic}
        """


rule GenerateSyntheticData:
    input:
        folder=rules.Train.output.decoded_folder,
        input_data=rules.preprocessing_ConcatImputedFiles.output.transformed,
        encodings=rules.GenerateSyntheticPatients.output.encodings,
        grouping=rules.preprocessing_Preprocessing.input.grouping,
        script="vambn/modelling/run_model.py",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z0-9]*",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    params:
        module=lambda x: x.module,
        cli_arg=lambda x: "gan_decode" if x.var == "wgan" else "ldecode",
        python_command="srun python" if config["snakemake"]["use_slurm"] else "python",
    output:
        output_data="{output_dir}/synthetic/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/{module}_data.csv",
    shell:
        """
        {params.python_command} -m vambn.modelling.run_model hivae {params.cli_arg} {wildcards.module} {input.folder} \
        {input.encodings} {input.grouping} {output.output_data}
        """


def get_decoded_data(wildcards):
    file_name = rules.preprocessing_Preprocessing.input.groups.format(**vars(wildcards))
    with open(file_name, "r") as f:
        groups = f.read().splitlines()

    modules = [f"{x}" for x in groups for i in range(1)]

    return [
        f"{wildcards.output_dir}/synthetic/traditional_{wildcards.dataset_name}_{wildcards.var}_{wildcards.mtl}/{module}_data.csv"
        for module in modules
    ]


rule MergeDecodedData:
    input:
        files=get_decoded_data,
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z0-9]*",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    output:
        output_data="{output_dir}/synthetic/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/data.csv",
    run:
        import pandas as pd
        from functools import reduce

        files = input["files"]
        dfs = [pd.read_csv(file) for file in files]

        # identify the common columns
        common_cols = reduce(
            lambda left, right: left.intersection(right),
            [set(df.columns) for df in dfs],
        )

        merged = reduce(
            lambda left, right: pd.merge(
                left, right, on=list(common_cols), how="outer"
            ),
            dfs,
        )

        merged.to_csv(output["output_data"])


rule GenerateSyndatData:
    input:
        original=rules.preprocessing_Preprocessing.input.data,
        decoded=rules.GatherDecodedData.output.output_data,
        synthetic=rules.MergeDecodedData.output.output_data,
        file="vambn/utils/syndat_conversion.py",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z0-9]*",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    output:
        raw_syndat="{output_dir}/syndat/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/raw_syndat.csv",
        decoded_syndat="{output_dir}/syndat/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/decoded_syndat.csv",
        synthetic_syndat="{output_dir}/syndat/{model_variant,traditional}_{dataset_name}_{var}_{mtl}/synthetic_syndat.csv",
    shell:
        """
        python {input.file} {input.original} {input.decoded} {input.synthetic} {output.raw_syndat} {output.decoded_syndat} {output.synthetic_syndat}
        """
