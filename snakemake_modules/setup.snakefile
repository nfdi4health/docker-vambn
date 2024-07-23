configfile: "vambn_config.yml"


rule all:
    input:
        "vambn_config.yml",
        expand(
            "{output_dir}/.r-setup-complete",
            output_dir=config["snakemake"]["output_dir"],
        ),

def setup_files(wildcards):
    if config["general"]["logging"]["mlflow"]["use"]:
        return [
            "r-dependencies.R",
            "snakemake_modules/setup.snakefile",
        ]
    else:
        return [
            "r-dependencies.R",
        ]

rule setupR:
    input:
        setup_files
    params:
        r_module=config["snakemake"]["cluster_modules"]["R"],
        r_env=config["snakemake"]["r_env"],
    output:
        "{output_dir}/.r-setup-complete",
    conda:
        config["snakemake"]["r_env"]
    shell:
        """
        if [ {params.r_module} != "None" ]; then
            module load {params.r_module}
        fi
        which Rscript
        sleep 5

        Rscript {input} && touch {output}
        """


rule initializeMlflowExperiment:
    params:
        mlflow_tracking_uri=config["general"]["logging"]["mlflow"]["tracking_uri"],
        mlflow_experiment_name=config["general"]["logging"]["mlflow"]["experiment_name"],
    output:
        "{output_dir}/.mlflow-experiment-initialized",
    run:
        import mlflow
        import datetime

        mlflow.set_tracking_uri(params.mlflow_tracking_uri)
        exp = mlflow.get_experiment_by_name(params.mlflow_experiment_name)
        if not exp:
            print(f"Creating new MLflow experiment: {params.mlflow_experiment_name}")
            mlflow.create_experiment(params.mlflow_experiment_name)
        else:
            print(f"MLflow experiment {params.mlflow_experiment_name} already exists")

        with open(output[0], "w") as f:
            # write when the experiment was initialized
            f.write("Initialized at: " + str(datetime.datetime.now()))
