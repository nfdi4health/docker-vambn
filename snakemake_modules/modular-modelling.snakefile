from glob import glob
import re
from typing import Set
import random
from itertools import combinations
from snakemake.utils import min_version
import sys

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
# Modular Modelling
################################################################################


# Rule to generate all possible outputs
rule all:
    input:
        expand(
            "{output_dir}/fit/modular_{shared}_{dataset_name}_{var}_{mtl}/overall_metrics.csv",
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
                "encoderMtl",
            ],
        ),
    default_target: True


# Optimize hyperparameters for modular models
rule Optimize:
    input:
        data=rules.preprocessing_Preprocessing.output.folder,
        config=config_path,
        script="vambn/modelling/run_model.py",
    params:
        experiment_name=lambda x: f"modular_{x.shared}_{x.dataset_name}_{x.var}_{x.mtl}_{config['general']['logging']['mlflow']['experiment_name']}",
        n_trials=config["optimization"]["n_modular_trials"],
        checkpoint_path="{output_dir}/optimization/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/checkpoints",
        database=(
            "{output_dir}/optimization/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/optuna.db"
            if config["general"]["optuna_db"] is None
            else config["general"]["optuna_db"]
        ),
        cli_arg=lambda x: "gan_optimize" if x.var == "wgan" else "optimize",
        python_command="srun python" if config["snakemake"]["use_slurm"] else "python",
    threads: 8
    log:
        base="logs/{output_dir}/opt_{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}.txt",
        stdout="logs/{output_dir}/opt_{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}.stdout",
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
    resources:
        runtime="48h",
        mem_mb=8000,
    output:
        parameter_file="{output_dir}/optimization/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/optimization_results.json",
    shell:
        """
        performed_trials=$(python -m vambn.utils.trial_counter {params.database} {params.experiment_name})
        trials_to_do=$(( {params.n_trials} - $performed_trials))
        echo "Performed $performed_trials"
        echo "Run $trials_to_do further trials"
        {params.python_command} -m vambn.modelling.run_model modular {params.cli_arg} {input.config} {wildcards.shared} {input.data} {params.experiment_name} {threads} {params.checkpoint_path} {params.database} \
        {output.parameter_file} --num-trials=$trials_to_do --log-file={log.base} > {log.stdout} 2>&1
        """


rule Train:
    input:
        data=rules.preprocessing_Preprocessing.output.folder,
        config=config_path,
        script="vambn/modelling/run_model.py",
        parameter_file=rules.Optimize.output.parameter_file,
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    params:
        checkpoint_path="{output_dir}/fit/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/checkpoints",
        cli_arg=lambda x: "gan_train" if x.var == "wgan" else "train",
        python_command="srun python" if config["snakemake"]["use_slurm"] else "python",
    threads: 8
    log:
        base="logs/{output_dir}/fit_{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}.txt",
        stdout="logs/{output_dir}/fit_{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}.stdout",
    resources:
        runtime="12h",
        mem_mb=8000,
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
        loss_mode="[A-Za-z]+",
    output:
        results="{output_dir}/fit/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/overall_metrics.csv",
        decoded_folder=directory(
            "{output_dir}/fit/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}"
        ),
        metaenc="{output_dir}/fit/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/data_outputs/meta_enc.csv",
        model_file="{output_dir}/fit/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/model.bin",
        trainer_file="{output_dir}/fit/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/trainer.pkl",
    shell:
        """
        {params.python_command} -m vambn.modelling.run_model modular {params.cli_arg} {input.config} {wildcards.shared} {input.data} {threads} \
        {params.checkpoint_path} {input.parameter_file} {output.decoded_folder} --log-file={log.base} > {log.stdout} 2>&1
        """


rule GatherDecodedData:
    input:
        decoded_folder=rules.Train.output.decoded_folder,
        stalone_data=rules.preprocessing_ConcatStaloneFiles.output.concat,
        script="vambn/data/make_data.py",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
        loss_mode="[A-Za-z]+",
    output:
        output_data="{output_dir}/decoded/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/data.csv",
    shell:
        """
        python -m vambn.data.make_data gather modular {input.decoded_folder} {input.stalone_data} {output.output_data}
        """


rule GenerateBayesianNetwork:
    input:
        stalone_data=rules.preprocessing_ConcatStaloneFiles.output.concat,
        metaenc=rules.Train.output.metaenc,
        grouping_file=rules.preprocessing_Preprocessing.input.grouping,
        blacklist=rules.preprocessing_Preprocessing.input.blacklist,
        whitelist=rules.preprocessing_Preprocessing.input.whitelist,
        start_dag=rules.preprocessing_Preprocessing.input.start_dag,
        script="vambn-r/bayesian_net.R",
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        variant="(?!(traditional))[a-zA-Z]+",
        output_dir="[A-Za-z0-9]+",
        loss_mode="[A-Za-z]+",
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
        bnl_folds=config["snakemake"]["bn"]["folds"],
        n_bootstrap=config["snakemake"]["bn"]["n_bootstrap"],
        seed=config["snakemake"]["bn"]["seed"],
        r_module=config["snakemake"]["cluster_modules"]["R"],
        r_env=config["snakemake"]["r_env"],
    output:
        output_dir=directory(
            "{output_dir}/bn/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/"
        ),
        bn_out="{output_dir}/bn/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/bn.rds",
        bootstrap_out="{output_dir}/bn/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/bootstrap_strength.csv",
        likelihood_real="{output_dir}/bn/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/likelihood_real.csv",
        bn_data="{output_dir}/bn/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/bn_data.rds",
    threads: 8
    conda:
        config["snakemake"]["r_env"]
    resources:
        runtime="12h",
        mem_mb=8000,
    shell:
        """
        if [ {params.r_module} != "None" ]; then
            module load {params.r_module}
        fi

        Rscript {input.script} \
        --stalone_data={input.stalone_data} \
        --grouping_file={input.grouping_file} \
        --metaenc={input.metaenc} \
        --blacklist={input.blacklist} \
        --whitelist={input.whitelist} \
        --start_dag={input.start_dag} \
        --bnl_maxp={params.bnl_maxp} \
        --n_bootstrap={params.n_bootstrap} \
        --seed={params.seed} \
        --output_dir={output.output_dir} \
        --bn_out={output.bn_out} \
        --likelihood_out={output.likelihood_real} \
        --bn_data={output.bn_data} \
        --cores={threads} 
        """


rule GenerateSyntheticPatients:
    input:
        fitted=rules.GenerateBayesianNetwork.output.bn_out,
        script="vambn-r/generate_syn_patients.R",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    params:
        n_patients=1000,
        r_module=config["snakemake"]["cluster_modules"]["R"],
        r_env=config["snakemake"]["r_env"],
    conda:
        config["snakemake"]["r_env"]
    output:
        encodings="{output_dir}/synthetic/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/synthetic_meta_enc.csv",
        likelihood_synthetic="{output_dir}/bn/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/likelihood_synthetic.csv",
    shell:
        """
        if [ {params.r_module} != "None" ]; then
            module load {params.r_module}
        fi

        Rscript {input.script} --fitted_model={input.fitted} --n_patients={params.n_patients} --encoding={output.encodings} --likelihood_out={output.likelihood_synthetic}
        """


rule GenerateSyntheticData:
    input:
        trainer=rules.Train.output.decoded_folder,
        input_data=rules.preprocessing_ConcatImputedFiles.output.transformed,
        encodings=rules.GenerateSyntheticPatients.output.encodings,
        grouping=rules.preprocessing_Preprocessing.input.grouping,
        groups=rules.preprocessing_Preprocessing.input.groups,
        script="vambn/modelling/run_model.py",
    params:
        cli_arg=lambda x: "gan_decode" if x.var == "wgan" else "decode",
        python_command="srun python" if config["snakemake"]["use_slurm"] else "python",
    threads: 1
    wildcard_constraints:
        dataset_name="[A-Za-z]*",
    output:
        output_data="{output_dir}/synthetic/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/data.csv",
    shell:
        """
        variable=$(tr '\n' ',' < {input.groups})
        {params.python_command} -m vambn.modelling.run_model modular {params.cli_arg} {input.trainer}  \
        {input.encodings} {input.grouping} {output.output_data} --modules-string="$variable"
        """


rule GenerateSyndatData:
    input:
        original=rules.preprocessing_Preprocessing.input.data,
        decoded=rules.GatherDecodedData.output.output_data,
        synthetic=rules.GenerateSyntheticData.output.output_data,
        file="vambn/utils/syndat_conversion.py",
    threads: 1
    wildcard_constraints:
        module="[a-zA-Z0-9]*",
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
        output_dir="[A-Za-z0-9]+",
        shared="[a-zA-Z0-9]+",
    output:
        raw_syndat="{output_dir}/syndat/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/raw_syndat.csv",
        decoded_syndat="{output_dir}/syndat/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/decoded_syndat.csv",
        synthetic_syndat="{output_dir}/syndat/{model_variant,modular}_{shared}_{dataset_name}_{var}_{mtl}/synthetic_syndat.csv",
    shell:
        """
        python {input.file} {input.original} {input.decoded} {input.synthetic} {output.raw_syndat} {output.decoded_syndat} {output.synthetic_syndat}
        """
