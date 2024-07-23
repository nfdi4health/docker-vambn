# Preprocessing module
from glob import glob


# Snakemake config and rule check
configfile: "vambn_config.yml"


if (
    config["snakemake"]["exclusive_dataset"] is not None
    and config["snakemake"]["excluded_datasets"] is not None
):
    raise Exception(
        "Only one of 'exclusive_dataset' and 'excluded_datasets' can be set."
    )


def obtain_dataset_wildcards():
    """Obtain names of suitable datasets"""
    available = glob_wildcards("data/raw/input_{dataset_name, [A-Za-z]+}.csv")[0]
    if (
        config["snakemake"]["exclusive_dataset"] is not None
        and config["snakemake"]["exclusive_dataset"] in available
    ):
        return [config["snakemake"]["exclusive_dataset"]]
    elif config["snakemake"]["excluded_datasets"] is not None:
        return [
            dataset
            for dataset in available
            if dataset not in config["snakemake"]["excluded_datasets"]
        ]
    else:
        return available


DATASETS = obtain_dataset_wildcards()
config["snakemake"]["DATASETS"] = DATASETS
# print(f"Working with the following datasets: {config['snakemake']['DATASETS']}")


module setupEnv:
    snakefile:
        "setup.snakefile"
    config:
        config


use rule * from setupEnv as setupEnv_*


rule all:
    input:
        expand(
            "{output_dir}/.r-setup-complete",
            output_dir=config["snakemake"]["output_dir"],
        ),
        expand(
            "{output_dir}/data/processed/{dataset_name}/split",
            dataset_name=DATASETS,
            output_dir=config["snakemake"]["output_dir"],
        ),
        expand(
            "{output_dir}/data/processed/{dataset_name}/concatenated{suffix}",
            suffix=["_raw.csv", "_imputed.csv", "_stalone.csv"],
            dataset_name=DATASETS,
            output_dir=config["snakemake"]["output_dir"],
        ),
    default_target: True


rule checkInputAvailability:
    input:
        data="data/raw/input_{dataset_name}.csv",
        grouping="data/raw/grouping_{dataset_name}.csv",
        groups="data/raw/groups_{dataset_name}.txt",
        blacklist="data/raw/blacklist_{dataset_name}.csv",
        whitelist="data/raw/whitelist_{dataset_name}.csv",
        start_dag="data/raw/startDag_{dataset_name}.csv",
        script="vambn/data/make_data.py",
        dependency_install=rules.setupEnv_all.input,
    output:
        indicator="{output_dir}/data/interim/avail_{dataset_name}",
    shell:
        """
        touch {output.indicator}
        """


rule Preprocessing:
    input:
        data=rules.checkInputAvailability.input.data,
        grouping=rules.checkInputAvailability.input.grouping,
        groups=rules.checkInputAvailability.input.groups,
        preprocessing_config="data/raw/config_{dataset_name}.json",
        blacklist=rules.checkInputAvailability.input.blacklist,
        whitelist=rules.checkInputAvailability.input.whitelist,
        start_dag=rules.checkInputAvailability.input.start_dag,
        script="vambn/data/make_data.py",
        dependency_install=rules.setupEnv_all.input,
    output:
        folder=directory("{output_dir}/data/processed/{dataset_name}/split"),
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
    threads: 1
    log:
        "logs/{output_dir}/preprocessing_{dataset_name}.txt",
    shell:
        """
        python -m vambn.data.make_data preprocessing make {input.data} {input.grouping} {input.groups} {input.preprocessing_config} {output.folder} --log-file={log}
        """


rule ConcatImputedFiles:
    input:
        folder=rules.Preprocessing.output.folder,
        script="vambn/data/make_data.py",
    output:
        concat="{output_dir}/data/processed/{dataset_name}/concatenated_imputed.csv",
        transformed="{output_dir}/data/processed/{dataset_name}/concatenated_imputed_transformed.csv",
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
    threads: 1
    log:
        "logs/{output_dir}/concat-imputed_{dataset_name}.txt",
    shell:
        """
        python -m vambn.data.make_data preprocessing merge-imputed-data {input.folder} {output}
        """


rule ConcatStaloneFiles:
    input:
        folder=rules.Preprocessing.output.folder,
        script="vambn/data/make_data.py",
    output:
        concat="{output_dir}/data/processed/{dataset_name}/concatenated_stalone.csv",
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
    threads: 1
    log:
        "logs/{output_dir}/concat-stalone_{dataset_name}.txt",
    shell:
        """
        python -m vambn.data.make_data preprocessing merge-stalone-data {input.folder} {output}
        """


rule ConcatRawFiles:
    input:
        folder=rules.Preprocessing.output.folder,
        script="vambn/data/make_data.py",
    output:
        concat="{output_dir}/data/processed/{dataset_name}/concatenated_raw.csv",
    wildcard_constraints:
        dataset_name="[A-Za-z]+",
        var="[A-Za-z]+",
        mtl="[A-Za-z]+",
    threads: 1
    log:
        "logs/{output_dir}/concat-raw_{dataset_name}.txt",
    shell:
        """
        python -m vambn.data.make_data preprocessing merge-raw-data {input.folder} {output}
        """
