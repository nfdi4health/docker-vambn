from pathlib import Path
from typing import Optional

import typer
import yaml
from pydantic import ValidationError
from pydantic.dataclasses import dataclass


@dataclass
class MlflowConfig:
    """
    Configuration class for MLflow settings.

    Attributes:
        use (bool): Whether to use MLflow for logging.
        tracking_uri (str): The URI of the MLflow tracking server.
        experiment_name (str): The name of the MLflow experiment.
    """

    use: bool
    tracking_uri: str
    experiment_name: str


@dataclass
class LoggingConfig:
    """
    Configuration class for logging settings.

    Attributes:
        level (int): The logging level.
        mlflow (MlflowConfig): The MLflow configuration.
    """

    level: int
    mlflow: MlflowConfig


@dataclass
class OptimizationConfig:
    """
    Configuration class for optimization settings.

    Attributes:
        max_epochs (int): The maximum number of epochs.
        early_stopping (bool): Whether to use early stopping during training.
        early_stopping_threshold (float): The threshold value for early stopping criterion.
        patience (int): The patience value for early stopping.
        folds (int): The number of folds for cross-validation.
        n_modular_trials (int): The number of trials for modular models.
        n_traditional_trials (int): The number of trials for traditional models.
        s_dim_lower (int): The lower bound of the s dimension.
        s_dim_upper (int): The upper bound of the s dimension.
        s_dim_step (int): The step size for the s dimension.
        fixed_s_dim (bool): Whether the s dimension is fixed.
        y_dim_lower (int): The lower bound of the y dimension.
        y_dim_upper (int): The upper bound of the y dimension.
        y_dim_step (int): The step size for the y dimension.
        fixed_y_dim (bool): Whether the y dimension is fixed.
        latent_dim_lower (int): The lower bound of the latent dimension.
        latent_dim_upper (int): The upper bound of the latent dimension.
        latent_dim_step (int): The step size for the latent dimension.
        batch_size_lower_n (int): The lower bound of the batch size.
        batch_size_upper_n (int): The upper bound of the batch size.
        learning_rate_lower (float): The lower bound of the learning rate.
        learning_rate_upper (float): The upper bound of the learning rate.
        fixed_learning_rate (bool): Whether the learning rate is fixed.
        lstm_layers_lower (int): The lower bound of the LSTM layers.
        lstm_layers_upper (int): The upper bound of the LSTM layers.
        lstm_layers_step (int): The step size for the LSTM layers.
        use_relative_correlation_error_for_optimization (bool): Whether to use relative correlation error for optimization.
        use_auc_for_optimization (bool): Whether to use AUC for optimization.
    """

    max_epochs: int
    early_stopping: bool
    early_stopping_threshold: float
    patience: int
    folds: int
    n_modular_trials: int
    n_traditional_trials: int
    s_dim_lower: int
    s_dim_upper: int
    s_dim_step: int
    fixed_s_dim: bool
    y_dim_lower: int
    y_dim_upper: int
    y_dim_step: int
    fixed_y_dim: bool
    latent_dim_lower: int
    latent_dim_upper: int
    latent_dim_step: int
    batch_size_lower_n: int
    batch_size_upper_n: int
    learning_rate_lower: float
    learning_rate_upper: float
    fixed_learning_rate: bool
    lstm_layers_lower: int
    lstm_layers_upper: int
    lstm_layers_step: int
    use_relative_correlation_error_for_optimization: bool
    use_auc_for_optimization: bool


@dataclass
class GeneralConfig:
    """Configuration for general settings.

    Attributes:
        seed: The random seed value.
        eval_batch_size: The batch size for evaluation.
        logging: The logging configuration.
        device: The device to run the computations on (e.g., 'cpu', 'cuda').
        optuna_db: Optional database string for Optuna.
    """

    seed: int
    eval_batch_size: int
    logging: LoggingConfig
    device: str
    optuna_db: Optional[str]


@dataclass
class TrainingConfig:
    """Configuration for training settings.

    Attributes:
        use_imputation_layer: Whether to use an imputation layer.
        use_mtl: Whether to use multi-task learning.
        with_gan: Whether to use a GAN.
    """

    use_imputation_layer: bool
    use_mtl: bool
    with_gan: bool


@dataclass
class PipelineConfig:
    """Configuration for the pipeline settings.

    Attributes:
        general: General configuration settings.
        optimization: Optimization configuration settings.
        training: Training configuration settings.
    """

    general: GeneralConfig
    optimization: OptimizationConfig
    training: TrainingConfig


def load_config_from_yaml(
    config_path: Path, use_mtl: bool, use_gan: bool
) -> PipelineConfig:  # noqa
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
        data["training"]["use_mtl"] = use_mtl
        data["training"]["with_gan"] = use_gan

    try:
        return PipelineConfig(**data)
    except ValidationError as e:
        print(f"Error validating config: {e}")
        exit(1)


def test(config_path: Path):
    config = load_config_from_yaml(config_path, False, False)
    print(config)


if __name__ == "__main__":
    typer.run(test)
