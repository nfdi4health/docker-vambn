import logging
import random
import re
from functools import reduce
from pathlib import Path
from typing import List, Optional

import mlflow
import optuna
import pandas as pd
import torch
import typer

import vambn.modelling.models.hivae.trainer as HiT
from vambn.data.helpers import load_vambn_data
from vambn.modelling.models.hivae.hivae import HivaeEncoding
from vambn.modelling.models.hivae.outputs import ModularHivaeEncoding
from vambn.utils.config import load_config_from_yaml
from vambn.utils.logging import setup_logging

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
app = typer.Typer(help="VAMBN Modelling")
modular_app = typer.Typer(help="Modular VAMBN 2.0")
traditional_app = typer.Typer(help="Traditional VAMBN")
traditional_vae = typer.Typer(help="Traditional VAE")
traditional_hivae = typer.Typer(help="Traditional HiVAE")
modular_vae = typer.Typer(help="Modular VAE 2.0")
app.add_typer(modular_app, name="modular")
app.add_typer(traditional_app, name="traditional")
app.add_typer(traditional_vae, name="vae")
app.add_typer(modular_vae, name="mvae")
app.add_typer(traditional_hivae, name="hivae")


def general_setup(config_path, workers, output_path, log_file):
    if not isinstance(log_file, Path):
        raise Exception("Log file must be a Path object")
    cfg = load_config_from_yaml(
        config_path=config_path,
        use_mtl="wmtl" in log_file.name,
        use_gan="wgan" in log_file.name,
    )
    if cfg.general.logging.mlflow.use:
        mlflow.set_tracking_uri(cfg.general.logging.mlflow.tracking_uri)
        mlflow.set_experiment(
            experiment_name=cfg.general.logging.mlflow.experiment_name
        )
    setup_logging(cfg.general.logging.level, log_file)
    torch.manual_seed(cfg.general.seed)
    random.seed(cfg.general.seed)
    torch.set_num_threads(workers)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    return cfg


def traditional_optimization(
    module_name,
    config_path,
    data_path,
    study_name,
    workers,
    checkpoint_path,
    study_path,
    parameter_file,
    num_trials,
    selected_visits,
    log_file,
    trainer_callable=HiT.TraditionalTrainer,
):
    cfg = general_setup(config_path, workers, None, log_file)
    dataset = load_vambn_data(
        data_folder=data_path,
        selected_visits=selected_visits,
    )
    trainer = trainer_callable(
        dataset=dataset,
        config=cfg,
        workers=workers,
        checkpoint_path=checkpoint_path,
        module_name=module_name,
        experiment_name=cfg.general.logging.mlflow.experiment_name,
        force_cpu=True if cfg.general.device == "cpu" else False,
    )
    if "sqlite" in str(study_path):
        study_path.parent.mkdir(parents=True, exist_ok=True)
    study_uri = (
        f"sqlite:///{str(study_path)}"
        if cfg.general.optuna_db is None
        else cfg.general.optuna_db
    )
    if cfg.optimization.use_relative_correlation_error_for_optimization:
        directions = ["minimize", "minimize"]
        metric_names = ["loss", "relative_correlation_error"]
    elif cfg.optimization.use_auc_for_optimization:
        directions = ["maximize"]
        metric_names = ["auc_quality"]
    else:
        directions = ["minimize"]
        metric_names = ["loss"]

    study = optuna.create_study(
        directions=directions,
        study_name=study_name,
        storage=study_uri,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.general.seed),
    )
    study.set_metric_names(metric_names)
    parameters = trainer.hyperopt(study, num_trials)
    parameters.write_to_json(parameter_file)


def traditional_training(
    module_name,
    config_path,
    data_path,
    workers,
    checkpoint_path,
    parameter_file,
    output_path,
    selected_visits,
    log_file,
    trainer_callable=HiT.TraditionalTrainer,
):
    cfg = general_setup(config_path, workers, output_path, log_file)
    dataset = load_vambn_data(
        data_folder=data_path,
        selected_visits=selected_visits,
        selected_modules=[module_name],
    )

    trainer = trainer_callable(
        dataset=dataset,
        config=cfg,
        workers=workers,
        checkpoint_path=checkpoint_path,
        module_name=module_name,
        experiment_name=cfg.general.logging.mlflow.experiment_name,
        force_cpu=True if cfg.general.device == "cpu" else False,
    )
    parameters = HiT.Hyperparameters.read_from_json(parameter_file)
    trainer.train(parameters)
    dl = trainer.get_dataloader(
        dataset, cfg.general.eval_batch_size, shuffle=False
    )
    trainer.evaluate(dl, output_path)
    trainer.save_trainer(output_path)


def traditional_decoding(
    module_name,
    trainer_file,
    synthetic_metaenc,
    grouping_file,
    output_data,
    log_file,
    logging_level,
    trainer_callable=HiT.TraditionalTrainer,
):
    setup_logging(logging_level, log_file)

    trainer = trainer_callable.load_trainer(trainer_file)
    # input_data = pd.read_csv(input_data_file, index_col=0)

    grouping = pd.read_csv(grouping_file)
    grouping = grouping.loc[
        grouping["technical_group_name"].isin([module_name])
        | grouping["technical_group_name"].str.match("stalone_"),
        :,
    ]

    # stalone_idx = grouping["technical_group_name"].str.match("stalone_")
    # vambn_grouping = grouping.loc[~stalone_idx, :]

    # Extract generated stalone data
    synthetic_metaenc = pd.read_csv(synthetic_metaenc)
    stalone_names = [x for x in synthetic_metaenc.columns if "SA_" in x]
    synthetic_stalone_data = synthetic_metaenc.loc[:, stalone_names]

    # get column names per module
    module_columns = [
        re.sub("_VIS[0-9]+", "", x) for x in trainer.model.colnames
    ]
    # Group the different encodings etc per data module; try to get s_encodings, if not available, use ones
    # get z_encodings
    z_columns = [
        x for x in synthetic_metaenc.columns if x.startswith(f"{module_name}_z")
    ]
    module_z_enc = (
        torch.tensor(synthetic_metaenc.loc[:, z_columns].values)
        .view(-1, len(z_columns))
        .float()
    )
    s_columns = [
        x for x in synthetic_metaenc.columns if x.startswith(f"{module_name}_s")
    ]
    module_s_enc = (
        torch.tensor(synthetic_metaenc.loc[:, s_columns].values)
        .view(-1, len(s_columns))
        .long()
    )

    hivae_enc = HivaeEncoding(
        s=module_s_enc,
        z=module_z_enc,
        module=module_name,
    )
    decoded_data = trainer.decode(hivae_enc, use_mode=False)
    decoded_data = decoded_data[module_name]

    # value is of shape (n_samples, time_points, n_features) or (n_samples, n_features)
    module_df_list = []
    if len(decoded_data.shape) == 3:
        for i, name in zip(range(decoded_data.shape[2]), module_columns):
            var_df = []
            for time_point in range(decoded_data.shape[1]):
                var_df.append(
                    pd.DataFrame(
                        {
                            "SUBJID": range(decoded_data.shape[0]),
                            "VISIT": time_point + 1,
                            name: list(decoded_data[:, time_point, i].numpy()),
                        }
                    )
                )
            module_df_list.append(pd.concat(var_df, axis=0))
    else:
        for i, name in zip(range(decoded_data.shape[1]), module_columns):
            module_df_list.append(
                pd.DataFrame(
                    {
                        "SUBJID": range(decoded_data.shape[0]),
                        "VISIT": 1,
                        name: list(decoded_data[:, i].numpy()),
                    }
                )
            )

    decoded_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["SUBJID", "VISIT"], how="outer"
        ),
        module_df_list,
    )

    # repeat stalone data for each time point and replace the VISIT column
    def _copy_stalone(df, visit):
        stalone_vis = df.copy()
        stalone_vis["VISIT"] = visit
        stalone_vis["SUBJID"] = range(stalone_vis.shape[0])
        return stalone_vis

    synthetic_stalone_data = pd.concat(
        [
            _copy_stalone(synthetic_stalone_data, i)
            for i in range(1, int(decoded_df["VISIT"].max()) + 1)
        ],
        axis=0,
    )
    synthetic_stalone_data.set_index(["SUBJID", "VISIT"], inplace=True)
    decoded_df.set_index(["SUBJID", "VISIT"], inplace=True)
    # sort columns from decoded data alphabetically
    decoded_df = decoded_df.reindex(sorted(decoded_df.columns), axis=1)

    merged = pd.concat([synthetic_stalone_data, decoded_df], axis=1)
    merged.columns = [x.replace("SA_", "") for x in merged.columns.to_list()]

    merged.to_csv(output_data)


def modular_optimization(
    config_path,
    modular_variant,
    data_path,
    study_name,
    workers,
    checkpoint_path,
    study_path,
    parameter_file,
    num_trials,
    selected_visits,
    selected_modules,
    log_file,
    trainer_callable=HiT.ModularTrainer,
):
    cfg = general_setup(config_path, workers, None, log_file)
    dataset = load_vambn_data(
        data_folder=data_path,
        selected_visits=selected_visits,
        selected_modules=selected_modules,
    )
    trainer = trainer_callable(
        dataset=dataset,
        config=cfg,
        workers=workers,
        checkpoint_path=checkpoint_path,
        experiment_name=cfg.general.logging.mlflow.experiment_name,
        force_cpu=True if cfg.general.device == "cpu" else False,
        shared_element=modular_variant,
    )
    trainer.device = (
        torch.device(cfg.general.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if "sqlite" in str(study_path):
        study_path.parent.mkdir(parents=True, exist_ok=True)
    study_uri = (
        f"sqlite:///{str(study_path)}"
        if cfg.general.optuna_db is None
        else cfg.general.optuna_db
    )

    if cfg.optimization.use_relative_correlation_error_for_optimization:
        directions = ["minimize", "minimize"]
        metric_names = ["loss", "relative_correlation_error"]
    elif cfg.optimization.use_auc_for_optimization:
        directions = ["maximize"]
        metric_names = ["auc_quality"]

    else:
        directions = ["minimize"]
        metric_names = ["loss"]

    study = optuna.create_study(
        directions=directions,
        study_name=study_name,
        storage=study_uri,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.general.seed),
    )
    study.set_metric_names(
        metric_names=metric_names,
    )
    # with torch.autograd.anomaly_mode.detect_anomaly():
    parameters = trainer.hyperopt(study, num_trials)
    parameters.write_to_json(parameter_file)


def modular_training(
    config_path,
    modular_variant,
    data_path,
    workers,
    checkpoint_path,
    parameter_file,
    output_path,
    selected_visits,
    selected_modules,
    log_file,
    trainer_callable=HiT.ModularTrainer,
):
    cfg = general_setup(config_path, workers, output_path, log_file)
    dataset = load_vambn_data(
        data_folder=data_path,
        selected_visits=selected_visits,
        selected_modules=selected_modules,
    )
    trainer = trainer_callable(
        dataset=dataset,
        config=cfg,
        workers=workers,
        checkpoint_path=checkpoint_path,
        experiment_name=cfg.general.logging.mlflow.experiment_name,
        force_cpu=True if cfg.general.device == "cpu" else False,
        shared_element=modular_variant,
    )
    trainer.device = (
        torch.device(cfg.general.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    parameters = HiT.Hyperparameters.read_from_json(parameter_file)
    if isinstance(parameters.learning_rate, float):
        lr_dict = {
            module_name: parameters.learning_rate
            for module_name in dataset.module_names
        }
        lr_dict["shared"] = parameters.learning_rate
        parameters.learning_rate = lr_dict

    trainer.train(parameters)
    dl = trainer.get_dataloader(
        dataset, cfg.general.eval_batch_size, shuffle=False
    )
    trainer.evaluate(dl, output_path)
    trainer.save_trainer(output_path)


def modular_decoding(
    trainer_file,
    synthetic_metaenc,
    grouping_file,
    output_data,
    modules_string,
    log_file,
    logging_level,
    trainer_callable=HiT.ModularTrainer,
):
    setup_logging(logging_level, log_file)
    # Ensure output directories exist
    output_data.parent.mkdir(exist_ok=True, parents=True)

    # Load model and data
    trainer = trainer_callable.load_trainer(trainer_file)

    # input_data = pd.read_csv(input_data_file, index_col=0)
    grouping = pd.read_csv(str(grouping_file))
    if modules_string is None:
        selected_modules = [
            x
            for x in grouping["technical_group_name"].unique()
            if "stalone" not in x
        ]
    else:
        selected_modules = modules_string.split(",")

    # ensure that only the selected modules and those starting with "stalone_" are used
    grouping = grouping.loc[
        grouping["technical_group_name"].isin(selected_modules)
        | grouping["technical_group_name"].str.match("stalone_"),
        :,
    ]
    stalone_idx = grouping["technical_group_name"].str.match("stalone_")
    vambn_grouping = grouping.loc[~stalone_idx, :]

    # Extract generated stalone data
    synthetic_metaenc = pd.read_csv(synthetic_metaenc)
    stalone_names = [x for x in synthetic_metaenc.columns if "SA_" in x]
    synthetic_stalone_data = synthetic_metaenc.loc[:, stalone_names]

    # Group the different encodings etc per data module
    encodings = []
    module_wise_columns = {}
    modules = vambn_grouping["technical_group_name"].unique()
    for module_name in modules:
        s_enc = torch.tensor(
            synthetic_metaenc.loc[:, f"{module_name}_s"].values
        ).view(-1, 1)
        if f"{module_name}_z" not in synthetic_metaenc.columns:
            # get all columns with _z[0-9]+ suffix
            z_columns = [
                x
                for x in synthetic_metaenc.columns
                if re.match(f"{module_name}_z[0-9]+", x)
            ]
            z_enc = torch.tensor(
                synthetic_metaenc.loc[:, z_columns].values
            ).view(-1, len(z_columns))
        else:
            z_enc = torch.tensor(
                synthetic_metaenc.loc[:, f"{module_name}_z"].values
            ).view(-1, 1)

        assert (
            s_enc.shape[0] == z_enc.shape[0]
        ), "S and Z encodings do not match"
        hivae_enc = HivaeEncoding(s=s_enc, z=z_enc, module=module_name)
        encodings.append(hivae_enc)

        if isinstance(trainer.model, HiT.ModularHivae):
            model_cols = trainer.model.module_models[module_name].colnames
        else:
            model_cols = trainer.model.model.module_models[module_name].colnames
        module_wise_columns[module_name] = tuple(
            re.sub(r"_VIS[0-9]+", "", x) for x in model_cols
        )
    modular_enc = ModularHivaeEncoding(encodings, modules=modules)
    decoded_data = trainer.decode(modular_enc, use_mode=False)

    decoded_dfs = []
    for key, sample_tensor in decoded_data.items():
        # value is of shape (n_samples, time_points, n_features) or (n_samples, n_features)
        module_df = []
        if len(sample_tensor.shape) == 3:
            assert sample_tensor.shape[2] == len(
                module_wise_columns[key]
            ), "Sample tensor and column names do not match"
            for i, name in zip(
                range(sample_tensor.shape[2]), module_wise_columns[key]
            ):
                var_df = []
                for time_point in range(sample_tensor.shape[1]):
                    var_df.append(
                        pd.DataFrame(
                            {
                                "SUBJID": range(sample_tensor.shape[0]),
                                "VISIT": time_point + 1,
                                name: list(
                                    sample_tensor[:, time_point, i].numpy()
                                ),
                            }
                        )
                    )
                col_df = pd.concat(var_df, axis=0)
                module_df.append(col_df)
        else:
            for i, name in zip(
                range(sample_tensor.shape[1]), module_wise_columns[key]
            ):
                assert sample_tensor.shape[1] == len(
                    module_wise_columns[key]
                ), "Sample tensor and column names do not match"
                module_df.append(
                    pd.DataFrame(
                        {
                            "SUBJID": range(sample_tensor.shape[0]),
                            "VISIT": 1,
                            name: list(sample_tensor[:, i].numpy()),
                        }
                    )
                )

        df = module_df[0]
        for i in range(1, len(module_df)):
            df = pd.merge(df, module_df[i], on=["SUBJID", "VISIT"], how="outer")

        decoded_dfs.append(df)

    decoded_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["SUBJID", "VISIT"], how="outer"
        ),
        decoded_dfs,
    )

    # repeat stalone data for each time point and replace the VISIT column
    def _copy_stalone(df, visit):
        stalone_vis = df.copy()
        stalone_vis["VISIT"] = visit
        stalone_vis["SUBJID"] = range(stalone_vis.shape[0])
        return stalone_vis

    synthetic_stalone_data = pd.concat(
        [
            _copy_stalone(synthetic_stalone_data, i)
            for i in range(1, int(decoded_df["VISIT"].max()) + 1)
        ],
        axis=0,
    )
    synthetic_stalone_data.set_index(["SUBJID", "VISIT"], inplace=True)
    decoded_df.set_index(["SUBJID", "VISIT"], inplace=True)
    # sort columns from decoded data alphabetically
    decoded_df = decoded_df.reindex(sorted(decoded_df.columns), axis=1)

    merged = pd.concat([synthetic_stalone_data, decoded_df], axis=1)
    merged.columns = [x.replace("SA_", "") for x in merged.columns.to_list()]

    merged.to_csv(output_data)


################################################################################
# CLI ##########################################################################
################################################################################


@traditional_hivae.command("loptimize")
def single_lhivae_optimize(
    module_name: str = typer.Argument(..., help="Name of the module"),
    config_path: Path = typer.Argument(..., help="Path to config file"),
    data_path: Path = typer.Argument(
        ..., help="Path to data folder (.../split/)"
    ),
    study_name: str = typer.Argument(..., help="Name of the study"),
    workers: int = typer.Argument(..., help="Number of workers"),
    checkpoint_path: Path = typer.Argument(..., help="Path to checkpoint file"),
    study_path: Path = typer.Argument(..., help="uri to the Optuna study"),
    parameter_file: Path = typer.Argument(..., help="Path to parameter file"),
    num_trials: Optional[int] = typer.Option(
        None, "--num-trials", "-n", help="Number of trials to run"
    ),
    selected_visits: Optional[List[int]] = typer.Option(
        None, "--selected-visits", "-v", help="Selected visits from the dataset"
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-l", help="Path to log file"
    ),
):
    traditional_optimization(
        module_name=module_name,
        config_path=config_path,
        data_path=data_path,
        study_name=study_name,
        workers=workers,
        checkpoint_path=checkpoint_path,
        study_path=study_path,
        parameter_file=parameter_file,
        num_trials=num_trials,
        selected_visits=selected_visits,
        log_file=log_file,
        trainer_callable=HiT.TraditionalTrainer,
    )


@traditional_hivae.command("gan_optimize")
def single_ganhivae_optimize(
    module_name: str = typer.Argument(..., help="Name of the module"),
    config_path: Path = typer.Argument(..., help="Path to config file"),
    data_path: Path = typer.Argument(
        ..., help="Path to data folder (.../split/)"
    ),
    study_name: str = typer.Argument(..., help="Name of the study"),
    workers: int = typer.Argument(..., help="Number of workers"),
    checkpoint_path: Path = typer.Argument(..., help="Path to checkpoint file"),
    study_path: Path = typer.Argument(..., help="uri to the Optuna study"),
    parameter_file: Path = typer.Argument(..., help="Path to parameter file"),
    num_trials: Optional[int] = typer.Option(
        None, "--num-trials", "-n", help="Number of trials to run"
    ),
    selected_visits: Optional[List[int]] = typer.Option(
        None, "--selected-visits", "-v", help="Selected visits from the dataset"
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-l", help="Path to log file"
    ),
):
    traditional_optimization(
        module_name=module_name,
        config_path=config_path,
        data_path=data_path,
        study_name=study_name,
        workers=workers,
        checkpoint_path=checkpoint_path,
        study_path=study_path,
        parameter_file=parameter_file,
        num_trials=num_trials,
        selected_visits=selected_visits,
        log_file=log_file,
        trainer_callable=HiT.TraditionalGanTrainer,
    )


@traditional_hivae.command("ltrain")
def single_lhivae_train(
    module_name: str = typer.Argument(..., help="Name of the module"),
    config_path: Path = typer.Argument(..., help="Path to config file"),
    data_path: Path = typer.Argument(
        ..., help="Path to data folder (.../split/)"
    ),
    workers: int = typer.Argument(..., help="Number of workers"),
    checkpoint_path: Path = typer.Argument(..., help="Path to checkpoint file"),
    parameter_file: Path = typer.Argument(..., help="Path to parameter file"),
    output_path: Path = typer.Argument(..., help="Path to output folder"),
    selected_visits: Optional[List[int]] = typer.Option(
        None, "--selected-visits", "-v", help="Selected visits from the dataset"
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-l", help="Path to log file"
    ),
):
    traditional_training(
        module_name=module_name,
        config_path=config_path,
        data_path=data_path,
        workers=workers,
        checkpoint_path=checkpoint_path,
        parameter_file=parameter_file,
        output_path=output_path,
        selected_visits=selected_visits,
        log_file=log_file,
    )


@traditional_hivae.command("gan_train")
def single_ganhivae_train(
    module_name: str = typer.Argument(..., help="Name of the module"),
    config_path: Path = typer.Argument(..., help="Path to config file"),
    data_path: Path = typer.Argument(
        ..., help="Path to data folder (.../split/)"
    ),
    workers: int = typer.Argument(..., help="Number of workers"),
    checkpoint_path: Path = typer.Argument(..., help="Path to checkpoint file"),
    parameter_file: Path = typer.Argument(..., help="Path to parameter file"),
    output_path: Path = typer.Argument(..., help="Path to output folder"),
    selected_visits: Optional[List[int]] = typer.Option(
        None, "--selected-visits", "-v", help="Selected visits from the dataset"
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-l", help="Path to log file"
    ),
):
    traditional_training(
        module_name=module_name,
        config_path=config_path,
        data_path=data_path,
        workers=workers,
        checkpoint_path=checkpoint_path,
        parameter_file=parameter_file,
        output_path=output_path,
        selected_visits=selected_visits,
        log_file=log_file,
        trainer_callable=HiT.TraditionalGanTrainer,
    )


@traditional_hivae.command("ldecode")
def single_lhivae_decode(
    module_name: str,
    trainer_file: Path,
    synthetic_metaenc: Path,
    grouping_file: Path,
    output_data: Path,
    log_file: Optional[Path] = None,
    logging_level: int = 20,
):
    return traditional_decoding(
        module_name=module_name,
        trainer_file=trainer_file,
        synthetic_metaenc=synthetic_metaenc,
        grouping_file=grouping_file,
        output_data=output_data,
        log_file=log_file,
        logging_level=logging_level,
        trainer_callable=HiT.TraditionalTrainer,
    )


@traditional_hivae.command("gan_decode")
def single_ganhivae_decode(
    module_name: str,
    trainer_file: Path,
    synthetic_metaenc: Path,
    grouping_file: Path,
    output_data: Path,
    log_file: Optional[Path] = None,
    logging_level: int = 20,
):
    return traditional_decoding(
        module_name=module_name,
        trainer_file=trainer_file,
        synthetic_metaenc=synthetic_metaenc,
        grouping_file=grouping_file,
        output_data=output_data,
        log_file=log_file,
        logging_level=logging_level,
        trainer_callable=HiT.TraditionalGanTrainer,
    )


@modular_app.command("optimize")
def mhivae_optimize(
    config_path: Path = typer.Argument(..., help="Path to config file"),
    modular_variant: str = typer.Argument(
        ...,
        help="Modular variant to use (none, simple, concat, all_concat, all_concat_indiv, concat_indiv)",
    ),
    data_path: Path = typer.Argument(
        ..., help="Path to data folder (.../split/)"
    ),
    study_name: str = typer.Argument(..., help="Name of the study"),
    workers: int = typer.Argument(..., help="Number of workers"),
    checkpoint_path: Path = typer.Argument(..., help="Path to checkpoint file"),
    study_path: Path = typer.Argument(..., help="uri to the Optuna study"),
    parameter_file: Path = typer.Argument(..., help="Path to parameter file"),
    num_trials: Optional[int] = typer.Option(
        None, "--num-trials", "-n", help="Number of trials to run"
    ),
    selected_visits: Optional[List[int]] = typer.Option(
        None, "--selected-visits", "-v", help="Selected visits from the dataset"
    ),
    selected_modules: Optional[List[str]] = typer.Option(
        None,
        "--selected-modules",
        "-m",
        help="Selected modules from the dataset",
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-l", help="Path to log file"
    ),
):
    modular_optimization(
        config_path=config_path,
        modular_variant=modular_variant,
        data_path=data_path,
        study_name=study_name,
        workers=workers,
        checkpoint_path=checkpoint_path,
        study_path=study_path,
        parameter_file=parameter_file,
        num_trials=num_trials,
        selected_visits=selected_visits,
        selected_modules=selected_modules,
        log_file=log_file,
        trainer_callable=HiT.ModularTrainer,
    )


@modular_app.command("gan_optimize")
def mganhivae_optimize(
    config_path: Path = typer.Argument(..., help="Path to config file"),
    modular_variant: str = typer.Argument(
        ...,
        help="Modular variant to use (none, simple, concat, all_concat, all_concat_indiv, concat_indiv)",
    ),
    data_path: Path = typer.Argument(
        ..., help="Path to data folder (.../split/)"
    ),
    study_name: str = typer.Argument(..., help="Name of the study"),
    workers: int = typer.Argument(..., help="Number of workers"),
    checkpoint_path: Path = typer.Argument(..., help="Path to checkpoint file"),
    study_path: Path = typer.Argument(..., help="uri to the Optuna study"),
    parameter_file: Path = typer.Argument(..., help="Path to parameter file"),
    num_trials: Optional[int] = typer.Option(
        None, "--num-trials", "-n", help="Number of trials to run"
    ),
    selected_visits: Optional[List[int]] = typer.Option(
        None, "--selected-visits", "-v", help="Selected visits from the dataset"
    ),
    selected_modules: Optional[List[str]] = typer.Option(
        None,
        "--selected-modules",
        "-m",
        help="Selected modules from the dataset",
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-l", help="Path to log file"
    ),
):
    modular_optimization(
        config_path=config_path,
        modular_variant=modular_variant,
        data_path=data_path,
        study_name=study_name,
        workers=workers,
        checkpoint_path=checkpoint_path,
        study_path=study_path,
        parameter_file=parameter_file,
        num_trials=num_trials,
        selected_visits=selected_visits,
        selected_modules=selected_modules,
        log_file=log_file,
        trainer_callable=HiT.ModularGanTrainer,
    )


@modular_app.command("train")
def mhivae_train(
    config_path: Path,
    modular_variant: str,
    data_path: Path,
    workers: int,
    checkpoint_path: Path,
    parameter_file: Path,
    output_path: Path,
    selected_visits: Optional[List[int]] = None,
    selected_modules: Optional[List[str]] = None,
    log_file: Optional[Path] = None,
):
    modular_training(
        config_path,
        modular_variant,
        data_path,
        workers,
        checkpoint_path,
        parameter_file,
        output_path,
        selected_visits,
        selected_modules,
        log_file,
        trainer_callable=HiT.ModularTrainer,
    )


@modular_app.command("gan_train")
def mganhivae_train(
    config_path: Path,
    modular_variant: str,
    data_path: Path,
    workers: int,
    checkpoint_path: Path,
    parameter_file: Path,
    output_path: Path,
    selected_visits: Optional[List[int]] = None,
    selected_modules: Optional[List[str]] = None,
    log_file: Optional[Path] = None,
):
    modular_training(
        config_path,
        modular_variant,
        data_path,
        workers,
        checkpoint_path,
        parameter_file,
        output_path,
        selected_visits,
        selected_modules,
        log_file,
        trainer_callable=HiT.ModularGanTrainer,
    )


@modular_app.command("decode")
def mhivae_decode(
    trainer_file: Path,
    synthetic_metaenc: Path,
    grouping_file: Path,
    output_data: Path,
    modules_string: Optional[str] = None,
    log_file: Optional[Path] = None,
    logging_level: int = 20,
):
    return modular_decoding(
        trainer_file,
        synthetic_metaenc,
        grouping_file,
        output_data,
        modules_string,
        log_file,
        logging_level,
        trainer_callable=HiT.ModularTrainer,
    )


@modular_app.command("gan_decode")
def mganhivae_decode(
    trainer_file: Path,
    synthetic_metaenc: Path,
    grouping_file: Path,
    output_data: Path,
    modules_string: Optional[str] = None,
    log_file: Optional[Path] = None,
    logging_level: int = 20,
):
    return modular_decoding(
        trainer_file,
        synthetic_metaenc,
        grouping_file,
        output_data,
        modules_string,
        log_file,
        logging_level,
        trainer_callable=HiT.ModularGanTrainer,
    )


if __name__ == "__main__":
    app()
