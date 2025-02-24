import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property, partial, reduce
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import dill
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import yaml
from pandas.core.api import DataFrame as DataFrame
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, train_test_split
from syndat.quality import get_auc
from torch._tensor import Tensor
from torch.utils.data import DataLoader

from vambn.data.dataclasses import VarTypes
from vambn.data.datasets import VambnDataset
from vambn.metrics.categorical import accuracy
from vambn.metrics.continous import nrmse
from vambn.metrics.jensen_shannon import jensen_shannon_distance
from vambn.metrics.relative_correlation import RelativeCorrelation
from vambn.modelling.models.config import DataModuleConfig
from vambn.modelling.models.hivae.config import HivaeConfig, ModularHivaeConfig
from vambn.modelling.models.hivae.gan_hivae import GanHivae, GanModularHivae
from vambn.modelling.models.hivae.hivae import Hivae, LstmHivae
from vambn.modelling.models.hivae.modular import ModularHivae
from vambn.modelling.models.hivae.outputs import (
    DecoderOutput,
    HivaeEncoding,
    HivaeOutput,
    ModularHivaeEncoding,
    ModularHivaeOutput,
)
from vambn.modelling.models.layers import ModifiedLinear
from vambn.utils.config import PipelineConfig
from vambn.utils.helpers import delete_directory

logger = logging.getLogger(__name__)

TConfig = TypeVar("TConfig")
TModel = TypeVar("TModel")
TPredict = TypeVar("TPredict")
TrainerType = TypeVar("TrainerType")
TEncoding = TypeVar("TEncoding")


def timed(fn: Callable) -> Tuple[Any, float]:
    """
    Decorator to time a function for benchmarking purposes.

    Args:
        fn (Callable): Function to be timed.

    Returns:
        Tuple[Any, float]: Result of the function and the time taken to execute it.
    """

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


@dataclass
class Hyperparameters:
    """
    Class representing the hyperparameters for the model trainer.

    Args:
        dim_s (int | Dict[str, int] | Tuple[int, ...]): Dimension(s) of the input sequence(s).
        dim_y (int | Dict[str, int] | Tuple[int, ...]): Dimension(s) of the output sequence(s).
        dim_z (int): Dimension of the latent space.
        dropout (float): Dropout rate.
        batch_size (int): Batch size.
        learning_rate (float | Dict[str, float] | Tuple[float, ...]): Learning rate(s).
        epochs (int): Number of training epochs.
        mtl_methods (Tuple[str, ...], optional): Multi-task learning methods. Defaults to ("identity",).
        lstm_layers (int, optional): Number of LSTM layers. Defaults to 1.
        dim_ys (Optional[int], optional): Dimension of the output sequence. Defaults to None.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
        early_stopping_threshold (float, optional): Threshold for early stopping. Defaults to 0.01.
        patience (int, optional): Number of epochs to wait before early stopping. Defaults to 10.

    Attributes:
        dim_s (int | Dict[str, int] | Tuple[int, ...]): Dimension(s) of the input sequence(s).
        dim_y (int | Dict[str, int] | Tuple[int, ...]): Dimension(s) of the output sequence(s).
        dim_z (int): Dimension of the latent space.
        dropout (float): Dropout rate.
        batch_size (int): Batch size.
        learning_rate (float | Dict[str, float] | Tuple[float, ...]): Learning rate(s).
        epochs (int): Number of training epochs.
        mtl_methods (Tuple[str, ...]): Multi-task learning methods.
        lstm_layers (int): Number of LSTM layers.
        dim_ys (Optional[int]): Dimension of the output sequence.
        early_stopping (bool): Whether to use early stopping.
        early_stopping_threshold (float): Threshold for early stopping.
        patience (int): Number of epochs to wait before early stopping.

    Methods:
        __post_init__(self): Post-initialization method.
        write_to_json(self, path: Path): Write the hyperparameters to a JSON file.
        read_from_json(cls, path: Path): Read the hyperparameters from a JSON file.

    """

    dim_s: int | Dict[str, int] | Tuple[int, ...]
    dim_y: int | Dict[str, int] | Tuple[int, ...]
    dim_z: int
    dropout: float
    batch_size: int
    learning_rate: float | Dict[str, float] | Tuple[float, ...]
    epochs: int
    mtl_methods: Tuple[str, ...] = ("identity",)
    lstm_layers: int = 1
    dim_ys: Optional[int] = None
    early_stopping: bool = False
    early_stopping_threshold: float = 0.01
    patience: int = 10

    def __post_init__(self):
        if isinstance(self.mtl_methods, List):
            self.mtl_methods = tuple(self.mtl_methods)
        elif isinstance(self.mtl_methods, str):
            self.mtl_methods = (self.mtl_methods,)

    def write_to_json(self, path: Path):
        """
        Write the hyperparameters to a JSON file.

        Args:
            path (Path): Path to the JSON file.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def read_from_json(cls, path: Path):
        """
        Read the hyperparameters from a JSON file.

        Args:
            path (Path): Path to the JSON file.

        Returns:
            Hyperparameters: An instance of the Hyperparameters class.

        """
        with path.open("r") as f:
            data = json.load(f)

        tmp = cls(**data)
        # FIXME: fix model script to avoid this step
        # check if , is in mtl_methods
        logger.info(tmp.mtl_methods)
        if len(tmp.mtl_methods) == 1 and "," in tmp.mtl_methods[0]:
            method_str = tmp.mtl_methods[0]
            tmp.mtl_methods = tuple(method_str.split(","))
        logger.info(tmp.mtl_methods)
        logger.debug(tmp)
        return tmp


class BaseTrainer(
    Generic[TConfig, TModel, TPredict, TrainerType, TEncoding], ABC
):
    """
    Base class for trainers in the VAMBN2 framework.

    Args:
        dataset (VambnDataset): The dataset to use for training.
        config (PipelineConfig): The configuration object for the pipeline.
        workers (int): The number of workers to use for data loading.
        checkpoint_path (Path): The path to save checkpoints during training.
        module_name (Optional[str], optional): The name of the module. Defaults to None.
        experiment_name (Optional[str], optional): The name of the experiment. Defaults to None.
        force_cpu (bool, optional): Whether to force CPU usage. Defaults to False.

    Attributes:
        dataset (VambnDataset): The dataset used for training.
        config (PipelineConfig): The configuration object for the pipeline.
        workers (int): The number of workers used for data loading.
        checkpoint_path (Path): The path to save checkpoints during training.
        model (Optional[TModel]): The model used for training.
        model_config (Optional[TConfig]): The configuration object for the model.
        module_name (Optional[str]): The name of the module.
        experiment_name (Optional[str]): The name of the experiment.
        type (str): The type of the trainer.
        device (torch.device): The device used for training.
        use_mtl (bool): Whether to use multi-task learning.
        use_gan (bool): Whether to use generative adversarial networks.
    """

    def __init__(
        self,
        dataset: VambnDataset,
        config: PipelineConfig,
        workers: int,
        checkpoint_path: Path,
        module_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        force_cpu: bool = False,
    ):
        self.dataset = dataset
        self.config = config
        self.workers = workers
        self.checkpoint_path = checkpoint_path

        self.model = None
        self.model_config = None
        self.module_name = module_name
        self.experiment_name = experiment_name
        self.type = "base"
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not force_cpu
            else torch.device("cpu")
        )
        self.use_mtl = config.training.use_mtl
        self.use_gan = config.training.with_gan
        logger.info(f"Use {self.device} for training")

    @cached_property
    def run_base(self) -> str:
        """
        Generates a base name for the training run. Used e.g. for MLflow run names.

        Returns:
            str: The base name for the Training run.
        """
        base = f"{self.type}_{'wmtl' if self.use_mtl else 'womtl'}_{'wgan' if self.use_gan else 'wogan'}"
        if self.module_name is not None:
            base += f"_{self.module_name}"
        return base

    def get_dataloader(
        self, dataset: VambnDataset, batch_size: int, shuffle: bool
    ) -> DataLoader:
        """
        Get a DataLoader object for the given dataset.

        Args:
            dataset (VambnDataset): The dataset to use.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: The DataLoader object.

        Raises:
            ValueError: If the dataset is empty.
        """
        # Set the number of workers to 0
        self.workers = 0

        # Create and return the DataLoader object
        return DataLoader(
            dataset.get_iter_dataset(self.module_name)
            if self.module_name is not None
            else dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            # collate_fn=self.custom_collate,
            persistent_workers=True if self.workers > 0 else False,
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True
            if shuffle and len(dataset) % batch_size <= 3
            else False,
        )

    def _set_device(self):
        """
        Sets the device for the model and dataset.

        This method sets the device for the model and dataset to the device specified in the `device` attribute.

        Returns:
            None
        """
        if self.model is not None:
            self.model.to(self.device)

    def multiple_objective_selection(
        self, study: optuna.Study, corr_weight: float = 0.8
    ) -> optuna.trial.FrozenTrial:
        """
        Selects the best trial from a given Optuna study based on multiple objectives.

        Args:
            study (optuna.Study): The Optuna study object.
            corr_weight (float, optional): The weight for the relative correlation error. Defaults to 0.8.

        Returns:
            optuna.trial.FrozenTrial: The best trial.

        Raises:
            ValueError: If no trials are found in the study.
        """

        # Get the best trials from the study
        best_trials = study.best_trials

        if not best_trials:
            raise ValueError("No trials found in the study.")

        # Calculate the weighted sum of relative correlation error and loss for each trial
        weighted_scores = []
        for trial in best_trials:
            corr_error = trial.values[1]
            loss = trial.values[0]
            weighted_score = corr_weight * corr_error + (1 - corr_weight) * loss
            weighted_scores.append(weighted_score)

        # Find the index of the trial with the minimum weighted score
        best_index = weighted_scores.index(min(weighted_scores))

        # Select the best trial based on the weighted score
        best_trial = best_trials[best_index]
        logger.info(f"Selected trial: {best_trial.number}")

        return best_trial

    def hyperopt(self, study: optuna.Study, num_trials: int) -> Hyperparameters:
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            study (optuna.Study): The Optuna study object.
            num_trials (int): The number of trials to run.

        Returns:
            Hyperparameters: The best hyperparameters found during optimization.

        Raises:
            ValueError: If no trials are found in the study.
        """
        with mlflow.start_run(run_name=f"{self.run_base}_hyperopt"):
            # Optimize the study
            study.optimize(self._objective, n_trials=num_trials)

            # Get the best trial parameters
            if self.config.optimization.use_relative_correlation_error_for_optimization:
                trial = self.multiple_objective_selection(study)
            else:
                trial = study.best_trial

            # Extract the best hyperparameters
            params = trial.params
            best_epoch = trial.user_attrs.get("best_epoch", None)
            params["epochs"] = best_epoch
            opt = self.config.optimization
            params["early_stopping"] = opt.early_stopping
            params["early_stopping_threshold"] = opt.early_stopping_threshold
            params["patience"] = opt.patience

            # Process the parameters
            params["batch_size"] = 2 ** params.pop("batch_size_n")
            if "hidden_dim_s" in params:
                params["dim_s"] = params.pop("hidden_dim_s")
            else:
                matching_keys = [k for k in params if "hidden_dim_s" in k]
                dim_s = {}
                for key in matching_keys:
                    module_name = key.split("_")[-1]
                    dim_s[module_name] = params.pop(key)
                params["dim_s"] = dim_s
            if "hidden_dim_y" in params:
                params["dim_y"] = params.pop("hidden_dim_y")
            else:
                matching_keys = [k for k in params if "hidden_dim_y_" in k]
                dim_y = {}
                for key in matching_keys:
                    module_name = key.split("_")[-1]
                    dim_y[module_name] = params.pop(key)
                params["dim_y"] = dim_y
            if "hidden_dim_ys" in params:
                params["dim_ys"] = params.pop("hidden_dim_ys")
            params["dim_z"] = params.pop("hidden_dim_z")
            if "learning_rate" not in params:
                matching_keys = [k for k in params if "learning_rate" in k]
                learning_rate = {}
                for key in matching_keys:
                    module_name = key.split("_")[-1]
                    learning_rate[module_name] = params.pop(key)
                params["learning_rate"] = learning_rate

            # Create Hyperparameters object
            hyperparameters = Hyperparameters(dropout=0.1, **params)

            # Log hyperparameters to MLflow
            mlflow.log_params(hyperparameters.__dict__)

        return hyperparameters

    def cleanup_checkpoints(self):
        """
        Cleans up the checkpoints directory.

        This method deletes the entire checkpoints directory specified by `self.checkpoint_path`.

        Returns:
            None
        """
        delete_directory(self.checkpoint_path)

    def optimize_model(self, model: Optional[TModel] = None) -> TModel:
        """
        Optimizes the model using a specified optimization function.

        Args:
            model (Optional[TModel], optional): The model to optimize. If None, the method optimizes self.model. Defaults to None.

        Returns:
            TModel: The optimized model.

        Notes:
            - The optimization function is specified by the opt_func variable.
            - The opt_func function should take a model as input and return an optimized model.
            - If model is None, the method optimizes self.model.
            - If model is not None, the method optimizes the specified model.

        Raises:
            TypeError: If the model is not of type TModel.

        """
        # Define the optimization function
        opt_func = partial(torch.compile, mode="reduce-overhead")
        # opt_func = lambda model: model

        if model is None:
            # Optimize self.model
            self.model = opt_func(model=self.model)
            return self.model
        else:
            return opt_func(model=model)

    def _add_encs(
        self,
        encodings_s: Dict[str, torch.Tensor],
        encodings_z: Dict[str, torch.Tensor],
        meta_enc: Dict[str, np.ndarray],
        decoder_output: DecoderOutput,
    ) -> None:
        """
        Helper function to prepare the encodings from the decoder output.

        This function takes the decoder output and adds the s and z encodings to the respective dictionaries.
        It also updates the meta encoding dictionary with the s and z encodings.

        Args:
            encodings_s (Dict[str, torch.Tensor]): Dictionary of s encodings per module.
            encodings_z (Dict[str, torch.Tensor]): Dictionary of z encodings per module.
            meta_enc (Dict[str, np.ndarray]): Entire meta encoding.
            decoder_output (DecoderOutput): The decoder output.

        Returns:
            None
        """
        dict_name = decoder_output.output_name

        # Add s and z encodings to the respective dictionaries
        if dict_name in encodings_s:
            encodings_s[dict_name].append(decoder_output.enc_s)
            encodings_z[dict_name].append(decoder_output.enc_z)
        else:
            encodings_s[dict_name] = [decoder_output.enc_s]
            encodings_z[dict_name] = [decoder_output.enc_z]

        # Update meta encoding dictionary with s and z encodings
        if decoder_output.enc_s.shape[1] > 1:
            s_dist = torch.argmax(decoder_output.enc_s, dim=1)
            s_name = f"{decoder_output.output_name}_s"
            if s_name in meta_enc:
                meta_enc[s_name] = np.concatenate(
                    [meta_enc[s_name], s_dist.numpy()]
                )
            else:
                meta_enc[s_name] = s_dist.numpy()
        for i in range(decoder_output.enc_z.shape[1]):
            z_name = f"{decoder_output.output_name}_z{i}"
            if z_name in meta_enc:
                meta_enc[z_name] = np.concatenate(
                    [meta_enc[z_name], decoder_output.enc_z[:, i].numpy()]
                )
            else:
                meta_enc[z_name] = decoder_output.enc_z[:, i].numpy()

    def save_model(self, path: Path) -> None:
        """
        Save the model and its configuration to the specified path.

        Args:
            path (Path): The path to save the model.

        Returns:
            None
        """
        self.save_model_config(path / "config.pkl")
        torch.save(self.model.state_dict(), path / "model.bin")

    def load_model(self, path: Path) -> None:
        """
        Load the model and its configuration from the specified path.

        Args:
            path (Path): The path to load the model from.

        Returns:
            None
        """
        self.read_model_config(path / "config.pkl")
        self.model = self.init_model(self.model_config)
        state_dict = torch.load(path / "model.bin")
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            state_dict = OrderedDict(
                {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            )
            self.model.load_state_dict(state_dict)

    @abstractmethod
    def save_model_config(self, path: Path) -> None:
        """
        Save the model configuration to the specified path.

        Args:
            path (Path): The path to save the model configuration.

        Returns:
            None
        """
        pass

    @abstractmethod
    def read_model_config(self, path: Path) -> None:
        """
        Read the model configuration from the specified path.

        Args:
            path (Path): The path to read the model configuration from.

        Returns:
            None
        """
        pass

    def cv_generator(
        self, splits: int, seed: int = 42
    ) -> Generator[Tuple[VambnDataset, VambnDataset], None, None]:
        """
        Generates train and validation datasets for cross-validation.

        Args:
            splits (int): Number of splits.
            seed (int, optional): Seed for split. Defaults to 42.

        Raises:
            ValueError: Number of splits must be greater than 0.

        Yields:
            Generator[Tuple[VambnDataset, VambnDataset], None, None]: A generator that yields tuples of train and validation datasets.
        """
        if splits < 1:
            raise ValueError("splits must be greater than 0")

        data_idx = np.arange(len(self.dataset))
        if splits == 1:
            train_idx, val_idx = train_test_split(
                data_idx, test_size=0.2, random_state=seed
            )
            yield (
                self.dataset.subset_by_idx(train_idx),
                self.dataset.subset_by_idx(val_idx),
            )
        else:
            cv = KFold(n_splits=splits, shuffle=True, random_state=seed)
            for train_idx, val_idx in cv.split(data_idx):
                if len(train_idx) == 0 or len(val_idx) == 0:
                    raise Exception("Empty split")
                yield (
                    self.dataset.subset_by_idx(train_idx),
                    self.dataset.subset_by_idx(val_idx),
                )

    @staticmethod
    def __lstm_forward_hook(module, input, output):
        # output is a tuple (output_tensor, (h_n, c_n))
        return output[0]  # We only want the output tensor for the next layer

    @staticmethod
    def setup_y_layer(
        z_dim: int, y_dim: int, dropout: float, n_layers: Optional[int] = None
    ) -> torch.nn.Module:
        if n_layers is None:
            return torch.nn.Sequential(
                # torch.nn.LayerNorm(z_dim),
                ModifiedLinear(z_dim, y_dim),
                torch.nn.Dropout(dropout),
            )
        else:
            layers = [
                # torch.nn.LayerNorm(z_dim),
                torch.nn.LSTM(
                    input_size=z_dim,
                    hidden_size=y_dim,
                    num_layers=n_layers,
                    dropout=dropout if n_layers > 1 else 0.0,
                    batch_first=True,
                ),
            ]

            layers[-1].register_forward_hook(BaseTrainer.__lstm_forward_hook)

            layers.append(torch.nn.Dropout(dropout))

            return torch.nn.Sequential(*layers)

    @abstractmethod
    def init_model(self, config: TConfig) -> TModel:
        pass

    @abstractmethod
    def train(self, best_parameters: Hyperparameters) -> TModel:
        pass

    def predict(self, dl: DataLoader | None) -> TPredict:
        if dl is None:
            data = self.dataset.get_iter_dataset(self.module_name)
            dl = self.get_dataloader(data, batch_size=128, shuffle=False)

        with torch.no_grad():
            return self.model.predict(dl)

    def decode(
        self,
        encoding: HivaeEncoding | ModularHivaeEncoding,
        use_mode: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        self.model.decoding = use_mode

        return {self.module_name: self.model.decode(encoding)}

    def save_trainer(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_model(path)
        self.model = None
        torch.save(self, path / "trainer.pkl", pickle_module=dill)

        with (path / "pipeline-config.yml").open("w") as f:
            f.write(yaml.dump(self.config.__dict__))

    @classmethod
    def load_trainer(cls, path: Path) -> TrainerType:
        trainer = torch.load(path / "trainer.pkl", pickle_module=dill)
        trainer.load_model(path)
        return trainer

    @abstractmethod
    def hyperparameters(self, trial: optuna.Trial) -> Hyperparameters:
        pass

    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float | Tuple[float, float]:
        pass

    def __handle_continous_data(
        self,
        original: pd.Series,
        decoded: pd.Series,
        output_file: Path,
        dtype: str,
    ) -> Tuple[float, float, float, float]:
        df = pd.DataFrame({"original": original, "decoded": decoded}).melt()
        sns.boxplot(x="variable", y="value", data=df)
        plt.savefig(output_file, dpi=300)
        plt.close()

        dec_tensor = torch.tensor(decoded.values)
        orig_tensor = torch.tensor(original.values)
        if orig_tensor.isnan().any() or orig_tensor.isinf().any():
            raise Exception("NaN values in original tensor")

        if dec_tensor.isnan().any() or dec_tensor.isinf().any():
            logger.warning("NaN values in decoded tensor")
            logger.warning(
                f"Found nan values: {dec_tensor.isnan().any()}, {dec_tensor.isnan().sum()}"
            )
            logger.warning(
                f"Found inf values: {dec_tensor.isinf().any()}, {dec_tensor.isinf().sum()}"
            )
            dec_tensor = dec_tensor.nan_to_num()
            dec_tensor[dec_tensor == float("inf")] = dec_tensor[
                dec_tensor != float("inf")
            ].max()
            logger.info(
                f"After replacing: {dec_tensor.isnan().any()}, {dec_tensor.isnan().sum()}"
            )
            logger.info(
                f"After replacing: {dec_tensor.isinf().any()}, {dec_tensor.isinf().sum()}"
            )

        error = float(
            nrmse(dec_tensor, orig_tensor, torch.ones_like(dec_tensor))
        )
        jsd = jensen_shannon_distance(dec_tensor, orig_tensor, dtype)
        try:
            statistic, pval = pearsonr(
                decoded.values.tolist(), original.values.tolist()
            )
        except ValueError:
            statistic, pval = 0.0, 1.0
            logger.warning("ValueError in pearsonr")
        return error, jsd, statistic, pval

    def handle_pos_data(
        self, original: pd.Series, decoded: pd.Series, output_file: Path
    ) -> Tuple[float, float, float, float]:
        return self.__handle_continous_data(
            original, decoded, output_file, "pos"
        )

    def handle_real_data(
        self, original: pd.Series, decoded: pd.Series, output_file: Path
    ) -> Tuple[float, float, float, float]:
        return self.__handle_continous_data(
            original, decoded, output_file, "real"
        )

    def handle_count_data(
        self, original: pd.Series, decoded: pd.Series, output_file: Path
    ) -> Tuple[float, float, float, float]:
        df = pd.DataFrame({"original": original, "decoded": decoded}).melt()
        sns.histplot(
            data=df, x="value", hue="variable", stat="probability", bins=30
        )
        plt.savefig(output_file, dpi=300)
        plt.close()

        dec_tensor = torch.tensor(decoded.values)
        orig_tensor = torch.tensor(original.values)

        nrmse_val = nrmse(dec_tensor, orig_tensor, torch.ones_like(dec_tensor))
        jsd = jensen_shannon_distance(dec_tensor, orig_tensor, "count")
        statistic, pval = pearsonr(
            decoded.astype(int).values, original.astype(int).values
        )
        return nrmse_val, jsd, statistic, pval

    def handle_categorical_data(
        self, original: pd.Series, decoded: pd.Series, output_file: Path
    ) -> Tuple[float, float, float, float]:
        df = pd.DataFrame({"original": original, "decoded": decoded}).melt()
        sns.histplot(data=df, x="value", hue="variable", stat="probability")
        plt.savefig(output_file, dpi=300)
        plt.close()

        dec_tensor = torch.tensor(decoded.values)
        orig_tensor = torch.tensor(original.values)

        error = accuracy(dec_tensor, orig_tensor, torch.ones_like(dec_tensor))
        jsd = jensen_shannon_distance(dec_tensor, orig_tensor, "cat")
        statistic, pval = spearmanr(
            decoded.values.tolist(), original.values.tolist()
        )
        return error, jsd, statistic, pval

    @abstractmethod
    def process_encodings(self, predictions: TPredict) -> TEncoding:
        raise NotImplementedError

    @staticmethod
    def reverse_scale(x: Tensor, variable_types: VarTypes) -> Tensor:
        copied_x = x.clone()
        for i, var_type in enumerate(variable_types):
            if var_type.data_type in ["real", "pos", "truncate_norm", "gamma"]:
                copied_x[:, i] = var_type.reverse_scale(copied_x[:, i])
        return copied_x

    def evaluate(self, dl: DataLoader, output_path: Path):
        output_path.mkdir(parents=True, exist_ok=True)

        predictions = self.predict(dl)
        encoding = self.process_encodings(predictions)

        data_output_path = output_path / "data_outputs"
        data_output_path.mkdir(exist_ok=True, parents=True)
        encoding.save_meta_enc(data_output_path / "meta_enc.csv")

        modules = (
            (self.module_name,)
            if self.module_name is not None
            else self.dataset.module_names
        )
        overall_metrics = [None] * len(modules)
        for k, module_name in enumerate(modules):
            submodules = self.dataset.get_modules(module_name)
            data, mask = self.dataset.get_longitudinal_data(module_name)
            sampled_data = encoding.get_samples(module_name)
            if sampled_data.ndim == 2 and data.ndim == 3:
                sampled_data = sampled_data.unsqueeze(1)

            if data.shape != sampled_data.shape:
                raise Exception("Data and sampled data have different shapes")

            if data.shape != mask.shape:
                raise Exception("Data and mask have different shapes")

            if data.ndim == 2:
                data = data.unsqueeze(1)
                sampled_data = sampled_data.unsqueeze(1)
                mask = mask.unsqueeze(1)

            num_timepoints = data.shape[1]
            decoded_data = [None] * num_timepoints
            original_data = [None] * num_timepoints
            mask_data = [None] * num_timepoints
            colnames = [
                re.sub("_VIS[0-9]+", "", c) for c in submodules[0].columns
            ]

            for time_point in range(num_timepoints):
                data_df = pd.DataFrame(
                    self.reverse_scale(
                        data[:, time_point, :], submodules[0].variable_types
                    ),
                    columns=colnames,
                )
                data_df["SUBJID"] = self.dataset.subj
                data_df["VISIT"] = time_point + 1
                data_df.set_index(["SUBJID", "VISIT"], inplace=True)
                original_data[time_point] = data_df

                sampled_data_df = pd.DataFrame(
                    sampled_data[:, time_point, :],
                    columns=colnames,
                )
                sampled_data_df["SUBJID"] = self.dataset.subj
                sampled_data_df["VISIT"] = time_point + 1
                sampled_data_df.set_index(["SUBJID", "VISIT"], inplace=True)
                decoded_data[time_point] = sampled_data_df

                mask_df = pd.DataFrame(mask[:, time_point, :], columns=colnames)
                mask_df["SUBJID"] = self.dataset.subj
                mask_df["VISIT"] = time_point + 1
                mask_df.set_index(["SUBJID", "VISIT"], inplace=True)
                mask_data[time_point] = mask_df

            original_data = pd.concat(original_data)
            decoded_data = pd.concat(decoded_data)

            # assert that we have the same visit ids in the original and decoded data
            assert original_data.index.equals(decoded_data.index), (
                f"Original and decoded data have different indices: "
                f"{original_data.index} != {decoded_data.index}"
            )
            if decoded_data.isna().any().any():
                raise Exception("NaN in decoded data")
            mask_data = pd.concat(mask_data)

            decoded_data.to_csv(data_output_path / f"{module_name}_decoded.csv")
            mask_data.to_csv(data_output_path / f"{module_name}_mask.csv")
            original_data.to_csv(
                data_output_path / f"{module_name}_original.csv"
            )

            # Calculate metrics per column
            error = [None] * len(colnames)
            jsd = [None] * len(colnames)
            correlation_stat = [None] * len(colnames)
            correlation_pval = [None] * len(colnames)

            distribution_path = output_path / "distributions"
            distribution_path.mkdir(parents=True, exist_ok=True)

            for i, column in enumerate(colnames):
                orig_col = original_data[column]
                decoded_col = decoded_data[column]
                mask_col = mask_data[column]

                orig_avail = orig_col[mask_col == 1]
                decoded_avail = decoded_col[mask_col == 1]

                col_type = submodules[0].variable_types[i].data_type
                if col_type in ["real", "pos", "truncate_norm", "gamma"]:
                    (
                        error[i],
                        jsd[i],
                        correlation_stat[i],
                        correlation_pval[i],
                    ) = self.handle_real_data(
                        orig_avail,
                        decoded_avail,
                        distribution_path / f"{module_name}_{column}.png",
                    )
                elif col_type == "count":
                    (
                        error[i],
                        jsd[i],
                        correlation_stat[i],
                        correlation_pval[i],
                    ) = self.handle_count_data(
                        orig_avail,
                        decoded_avail,
                        distribution_path / f"{module_name}_{column}.png",
                    )
                elif col_type == "cat":
                    (
                        error[i],
                        jsd[i],
                        correlation_stat[i],
                        correlation_pval[i],
                    ) = self.handle_categorical_data(
                        orig_avail,
                        decoded_avail,
                        distribution_path / f"{module_name}_{column}.png",
                    )
                else:
                    raise ValueError(f"Unknown data type {col_type}")

            overall_metrics[k] = pd.DataFrame(
                {
                    "module_name": [module_name] * len(colnames),
                    "column": colnames,
                    "error": [float(x) for x in error],
                    "jsd": jsd,
                    "correlation_stat": correlation_stat,
                    "correlation_pval": correlation_pval,
                }
            )

        overall_metrics = pd.concat(overall_metrics)
        overall_metrics.to_csv(output_path / "overall_metrics.csv", index=False)


GenericHivaeConfig = TypeVar("GenericHivaeConfig", bound=HivaeConfig)
GenericHivaeModel = TypeVar("GenericHivaeModel", bound=Hivae | LstmHivae)

GenericMHivaeConfig = TypeVar("GenericMHivaeConfig", bound=ModularHivaeConfig)
GenericMHivaeModel = TypeVar("GenericMHivaeModel", bound=ModularHivae)


class TraditionalTrainer(
    BaseTrainer[
        GenericHivaeConfig,
        GenericHivaeModel,
        HivaeOutput,
        "TraditionalTrainer",
        HivaeEncoding,
    ]
):
    def __init__(
        self,
        dataset: VambnDataset,
        config: PipelineConfig,
        workers: int,
        checkpoint_path: Path,
        module_name: str | None = None,
        experiment_name: str | None = None,
        force_cpu: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            config=config,
            workers=workers,
            checkpoint_path=checkpoint_path,
            module_name=module_name,
            experiment_name=experiment_name,
            force_cpu=force_cpu,
        )
        self.type = "traditional"
        if module_name is None:
            raise ValueError("Module name must be specified")
        elif module_name not in self.dataset.module_names:
            raise ValueError(
                f"Module name {module_name} not in dataset, available modules: {self.dataset.module_names}"
            )

    def process_encodings(self, predictions: HivaeOutput) -> HivaeEncoding:
        return HivaeEncoding(
            z=predictions.enc_z,
            s=predictions.enc_s,
            module=self.module_name,
            samples=predictions.samples,
            subjid=self.dataset.subj,
        )

    def init_model(self, config: GenericHivaeConfig) -> GenericHivaeModel:
        if config.is_longitudinal:
            return LstmHivae(
                n_layers=config.n_layers,
                num_timepoints=config.num_timepoints,
                dim_s=config.dim_s,
                dim_y=config.dim_y,
                dim_z=config.dim_z,
                input_dim=config.input_dim,
                module_name=self.module_name,
                mtl_method=config.mtl_methods,
                use_imputation_layer=config.use_imputation_layer,
                variable_types=config.variable_types,
                individual_model=True,
            )
        else:
            return Hivae(
                variable_types=config.variable_types,
                input_dim=config.input_dim,
                dim_s=config.dim_s,
                dim_y=config.dim_y,
                dim_z=config.dim_z,
                module_name=self.module_name,
                mtl_method=config.mtl_methods,
                use_imputation_layer=config.use_imputation_layer,
                individual_model=True,
            )

    def hyperparameters(self, trial: optuna.Trial) -> Hyperparameters:
        """
        Function to suggest hyperparameters for the model

        Args:
            trial (optuna.Trial): Trial instance

        Returns:
            Dict[str, Any]: Suggested hyperparameters
        """
        opt = self.config.optimization
        dim_s = trial.suggest_int(
            "hidden_dim_s",
            opt.s_dim_lower,
            opt.s_dim_upper,
            opt.s_dim_step,
        )
        dim_y = trial.suggest_int(
            "hidden_dim_y",
            opt.y_dim_lower,
            opt.y_dim_upper,
            opt.y_dim_step,
        )
        dim_z = trial.suggest_int(
            "hidden_dim_z",
            opt.latent_dim_lower,
            opt.latent_dim_upper,
            opt.latent_dim_step,
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            opt.learning_rate_lower,
            opt.learning_rate_upper,
            log=True,
        )
        batch_size_n = trial.suggest_int(
            "batch_size_n", opt.batch_size_lower_n, opt.batch_size_upper_n
        )
        batch_size = 2**batch_size_n
        # epochs = trial.suggest_int(
        #     "epochs",
        #     low=opt.epoch_lower,
        #     high=opt.epoch_upper,
        #     step=opt.epoch_step,
        # )

        if self.dataset.is_longitudinal:
            lstm_layers = trial.suggest_int(
                "lstm_layers",
                opt.lstm_layers_lower,
                opt.lstm_layers_upper,
                opt.lstm_layers_step,
            )
        else:
            lstm_layers = 1
        if self.use_mtl:
            mtl_string = trial.suggest_categorical(
                "mtl_methods",
                [
                    "gradnorm",
                    "graddrop",
                    "gradnorm,graddrop",
                ],
            )
            mtl_methods = (
                tuple(mtl_string.split(","))
                if "," in mtl_string
                else tuple([mtl_string])
            )
        else:
            mtl_methods = ("identity",)
        return Hyperparameters(
            dim_s=dim_s,
            dim_y=dim_y,
            dim_z=dim_z,
            dropout=0.1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=opt.max_epochs,
            early_stopping=opt.early_stopping,
            early_stopping_threshold=opt.early_stopping_threshold,
            patience=opt.patience,
            lstm_layers=lstm_layers,
            mtl_methods=mtl_methods,
        )

    def train(self, best_parameters: Hyperparameters) -> GenericHivaeModel:
        """
        Trains the model using the best hyperparameters.

        Args:
            best_parameters (Hyperparameters): The best hyperparameters obtained from optimization.

        Returns:
            GenericHivaeModel: The trained model.
        """
        ref_module = self.dataset.get_modules(self.module_name)[0]
        whole_dataloader = self.get_dataloader(
            self.dataset, best_parameters.batch_size, shuffle=True
        )
        self.model_config = HivaeConfig(
            name=self.module_name,
            variable_types=ref_module.variable_types,
            dim_s=best_parameters.dim_s,
            dim_y=best_parameters.dim_y,
            dim_z=best_parameters.dim_z,
            mtl_methods=best_parameters.mtl_methods,
            use_imputation_layer=self.config.training.use_imputation_layer,
            dropout=best_parameters.dropout,
            n_layers=best_parameters.lstm_layers,
            num_timepoints=self.dataset.visits_per_module[self.module_name],
        )
        self.model = self.init_model(self.model_config).to(self.device)
        self.model.device = self.device
        self.optimize_model()
        with mlflow.start_run(run_name=f"{self.run_base}_final_fit"):
            mlflow.log_params(best_parameters.__dict__)
            self.model.fit(
                train_dataloader=whole_dataloader,
                num_epochs=best_parameters.epochs,
                learning_rate=best_parameters.learning_rate,
                early_stopping=False,
            )
        return self.model

    def _objective(self, trial: optuna.Trial) -> float | Tuple[float, float]:
        """
        Objective function for the optimization process.

        Args:
            trial (optuna.Trial): The trial object for hyperparameter optimization.

        Returns:
            float or Tuple[float, float]: The loss value or a tuple of loss values.
        """
        trial_params = self.hyperparameters(trial)
        logger.info(f"Trial parameters: {trial_params}")
        fold_loss = []
        rel_corr_loss = []
        auc_metric = []
        ref_module = self.dataset.get_modules(self.module_name)[0]
        n_epochs = []

        for i, (train_data, val_data) in enumerate(
            self.cv_generator(self.config.optimization.folds)
        ):
            train_dataloader = self.get_dataloader(
                train_data, trial_params.batch_size, shuffle=True
            )
            val_dataloader = self.get_dataloader(
                val_data, trial_params.batch_size, shuffle=False
            )

            model_config = HivaeConfig(
                name=self.module_name,
                variable_types=ref_module.variable_types,
                dim_s=trial_params.dim_s,
                dim_y=trial_params.dim_y,
                dim_z=trial_params.dim_z,
                mtl_methods=trial_params.mtl_methods,
                use_imputation_layer=self.config.training.use_imputation_layer,
                dropout=trial_params.dropout,
                n_layers=trial_params.lstm_layers,
                num_timepoints=self.dataset.visits_per_module[self.module_name],
            )
            raw_model = self.init_model(model_config)
            raw_model.device = self.device
            model = self.optimize_model(raw_model)
            model.to(self.device)
            with mlflow.start_run(
                run_name=f"{self.run_base}_T{trial._trial_id}_F{i}",
                nested=True,
            ):
                mlflow.log_params(trial_params.__dict__)
                start = time.time()
                try:
                    fit_loss, fit_epoch = model.fit(
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        num_epochs=trial_params.epochs,
                        learning_rate=trial_params.learning_rate,
                        early_stopping=trial_params.early_stopping,
                        early_stopping_threshold=trial_params.early_stopping_threshold,
                        patience=trial_params.patience,
                    )
                except ValueError:
                    raise optuna.TrialPruned(
                        "Trial pruned due to error during training. Likely due to unsuitable hyperparameters"
                    )

                if ((time.time() - start) / 3600) > 2:
                    logger.warning(
                        f"Trial pruned due to very long execution ({(time.time() - start) / 3600}h at fold {i})"
                    )
                    raise optuna.TrialPruned()
                output = model.predict(val_dataloader)

                mlflow.log_metric("Final loss", output.avg_loss)
                logger.info(f"Fold {i} loss: {output.avg_loss}")
                fold_loss.append(output.avg_loss)
                n_epochs.append(fit_epoch)

                if (
                    self.config.optimization.use_relative_correlation_error_for_optimization
                    or self.config.optimization.use_auc_for_optimization
                ):
                    # Calculate the relative correlation loss
                    orig_data = val_data.to_pandas(module_name=self.module_name)
                    # convert decoded samples into a pandas dataframe
                    decoded_samples = output.samples
                    assert decoded_samples.shape[-1] == (
                        orig_data.shape[-1] - 2
                    )
                    column_names = [
                        re.sub(r"_VIS[0-9]+", "", c)
                        for c in self.dataset.get_modules(self.module_name)[
                            0
                        ].columns
                    ]
                    if decoded_samples.ndim == 2:
                        synthetic_data = pd.DataFrame(
                            decoded_samples.numpy(),
                            columns=column_names,
                        )
                        synthetic_data["SUBJID"] = val_data.subj
                        synthetic_data["VISIT"] = 1
                    else:
                        synthetic_data = pd.concat(
                            [
                                pd.DataFrame(
                                    decoded_samples[:, i, :].numpy(),
                                    columns=column_names,
                                )
                                for i in range(decoded_samples.shape[1])
                            ]
                        )
                        synthetic_data["SUBJID"] = (
                            val_data.subj
                            * self.dataset.visits_per_module[self.module_name]
                        )
                        synthetic_data["VISIT"] = np.repeat(
                            np.arange(1, decoded_samples.shape[1] + 1),
                            len(val_data),
                        )

                    synthetic_data_filtered = synthetic_data.loc[
                        :, orig_data.columns
                    ].drop(columns=["SUBJID", "VISIT"])
                    orig_data_filtered = orig_data.drop(
                        columns=["SUBJID", "VISIT"]
                    )

                    fold_rel_corr_loss, m1, m2 = RelativeCorrelation.error(
                        real=orig_data_filtered,
                        synthetic=synthetic_data_filtered,
                    )

                    auc = get_auc(orig_data_filtered, synthetic_data_filtered)
                    if auc < 0.5:
                        auc = 1 - auc
                    auc_quality = max(math.floor((1 - auc) * 200), 1)
                    mlflow.log_metric("AUC", auc)
                    mlflow.log_metric("AUC quality", auc_quality)
                    auc_metric.append(auc_quality)

                    mlflow.log_metric(
                        "Relative correlation loss", fold_rel_corr_loss
                    )
                    rel_corr_loss.append(fold_rel_corr_loss)

        loss = np.mean(fold_loss)
        avg_auc_metric = np.mean(auc_metric)

        # Get the n_epochs that correspond to the best loss
        best_epoch = n_epochs[np.argmin(fold_loss)]
        mlflow.log_metric("Best epoch", best_epoch)
        # Log the value in the trial
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("n_epochs", n_epochs)

        if self.config.optimization.use_relative_correlation_error_for_optimization:
            rel_corr_loss = np.mean(rel_corr_loss)
            logger.info(f"Trial loss: {loss}; rel_corr_loss: {rel_corr_loss}")
            return loss, rel_corr_loss
        elif self.config.optimization.use_auc_for_optimization:
            return avg_auc_metric
        else:
            logger.info(f"Trial loss: {loss}")
            return loss

    def save_model_config(self, path: Path):
        """
        Saves the model configuration to a file.

        Args:
            path (Path): The path to save the model configuration.
        """
        if self.model is None:
            raise Exception("Model should not be none")

        if self.model_config is None:
            raise Exception("Model config should not be none")

        with path.open("wb") as f:
            dill.dump(self.model_config, f)

    def read_model_config(self, path: Path):
        """
        Reads the model configuration from a file.

        Args:
            path (Path): The path to read the model configuration from.
        """
        with path.open("rb") as f:
            self.model_config = dill.load(f)


class TraditionalGanTrainer(TraditionalTrainer[HivaeConfig, GanHivae]):
    def init_model(self, config: HivaeConfig) -> GanHivae:
        """
        Initializes the GAN-HIVAE model.

        Args:
            config (HivaeConfig): The configuration for the GAN-HIVAE model.

        Returns:
            GanHivae: The initialized GAN-HIVAE model.
        """
        return GanHivae(
            variable_types=config.variable_types,
            input_dim=config.input_dim,
            dim_s=config.dim_s,
            dim_y=config.dim_y,
            dim_z=config.dim_z,
            module_name=config.name,
            mtl_method=config.mtl_methods,
            use_imputation_layer=config.use_imputation_layer,
            individual_model=True,
            n_layers=config.n_layers,
            num_timepoints=config.num_timepoints,
            noise_size=10,
        )


class ModularTrainer(
    BaseTrainer[
        GenericMHivaeConfig,
        GenericMHivaeModel,
        ModularHivaeOutput,
        "ModularTrainer",
        ModularHivaeEncoding,
    ]
):
    def __init__(
        self,
        dataset: VambnDataset,
        config: PipelineConfig,
        workers: int,
        checkpoint_path: Path,
        module_name: str | None = None,
        experiment_name: str | None = None,
        force_cpu: bool = False,
        shared_element: str = "none",
    ):
        """
        Initialize the ModularTrainer class.

        Args:
            dataset: The VambnDataset object.
            config: The PipelineConfig object.
            workers: The number of workers for data loading.
            checkpoint_path: The path to save checkpoints.
            module_name: The name of the module.
            experiment_name: The name of the experiment.
            force_cpu: Whether to force CPU usage.
            shared_element: The type of shared element.
        """
        super().__init__(
            dataset=dataset,
            config=config,
            workers=workers,
            checkpoint_path=checkpoint_path,
            module_name=module_name,
            experiment_name=experiment_name,
            force_cpu=force_cpu,
        )
        self.type = "modular"
        self.shared_element = shared_element

    def process_encodings(
        self, predictions: ModularHivaeOutput
    ) -> ModularHivaeEncoding:
        """
        Process the model predictions and return the encodings.

        Args:
            predictions: The model predictions.

        Returns:
            The ModularHivaeEncoding object.
        """
        return ModularHivaeEncoding(
            encodings=tuple(
                HivaeEncoding(
                    z=pred.enc_z,
                    s=pred.enc_s,
                    module=module_name,
                    samples=pred.samples,
                    subjid=self.dataset.subj,
                )
                for module_name, pred in zip(
                    self.dataset.module_names, predictions
                )
            ),
            modules=self.dataset.module_names,
        )

    def init_model(self, config: ModularHivaeConfig) -> ModularHivae:
        """
        Initialize the ModularHivae model.

        Args:
            config: The ModularHivaeConfig object.

        Returns:
            The initialized ModularHivae model.
        """
        return ModularHivae(
            dim_s=config.dim_s,
            dim_ys=config.dim_ys,
            dim_y=config.dim_y,
            dim_z=config.dim_z,
            module_config=config.module_config,
            mtl_method=config.mtl_method,
            shared_element_type=config.shared_element,
            use_imputation_layer=config.use_imputation_layer,
        )

    def hyperparameters(self, trial: optuna.Trial) -> Hyperparameters:
        """
        Generate hyperparameters for the trial.

        Args:
            trial: The optuna.Trial object.

        Returns:
            The generated Hyperparameters object.
        """
        opt = self.config.optimization
        if opt.fixed_s_dim:
            # Use fixed dimension for s
            fixed_dim_s = trial.suggest_int(
                "hidden_dim_s",
                opt.s_dim_lower,
                opt.s_dim_upper,
                opt.s_dim_step,
            )
            dim_s = {
                module_name: fixed_dim_s
                for module_name in self.dataset.module_names
            }
        else:
            # Use different dimensions for s
            dim_s = {
                module_name: trial.suggest_int(
                    f"hidden_dim_s_{module_name}",
                    opt.s_dim_lower,
                    opt.s_dim_upper,
                    opt.s_dim_step,
                )
                for module_name in self.dataset.module_names
            }
        if opt.fixed_y_dim:
            # Use fixed dimension for y
            fixed_y = trial.suggest_int(
                "hidden_dim_y",
                opt.y_dim_lower,
                opt.y_dim_upper,
                opt.y_dim_step,
            )
            dim_y = {module_name: fixed_y for module_name in dim_s}
        else:
            # Use different dimensions for y
            dim_y = {
                module_name: trial.suggest_int(
                    f"hidden_dim_y_{module_name}",
                    opt.y_dim_lower,
                    opt.y_dim_upper,
                    opt.y_dim_step,
                )
                for module_name in self.dataset.module_names
            }
        dim_z = trial.suggest_int(
            "hidden_dim_z",
            opt.latent_dim_lower,
            opt.latent_dim_upper,
            opt.latent_dim_step,
        )
        dim_ys = trial.suggest_int(
            "hidden_dim_ys",
            opt.y_dim_lower,
            opt.y_dim_upper,
            opt.y_dim_step,
        )
        if opt.fixed_learning_rate:
            lr = trial.suggest_float(
                "learning_rate",
                opt.learning_rate_lower,
                opt.learning_rate_upper,
                log=True,
            )
            learning_rate = {
                module_name: lr for module_name in self.dataset.module_names
            }
            learning_rate["learning_rate_shared"] = lr
        else:
            learning_rate = {
                module_name: trial.suggest_float(
                    f"learning_rate_{module_name}",
                    opt.learning_rate_lower,
                    opt.learning_rate_upper,
                    log=True,
                )
                for module_name in self.dataset.module_names
            }
            learning_rate["learning_rate_shared"] = trial.suggest_float(
                "learning_rate_shared",
                opt.learning_rate_lower,
                opt.learning_rate_upper,
                log=True,
            )

        batch_size_n = trial.suggest_int(
            "batch_size_n", opt.batch_size_lower_n, opt.batch_size_upper_n
        )
        batch_size = 2**batch_size_n

        if self.dataset.is_longitudinal:
            lstm_layers = trial.suggest_int(
                "lstm_layers",
                opt.lstm_layers_lower,
                opt.lstm_layers_upper,
                opt.lstm_layers_step,
            )
        else:
            lstm_layers = 1
        if self.use_mtl:
            mtl_string = trial.suggest_categorical(
                "mtl_methods",
                [
                    "gradnorm",
                    "graddrop",
                    "gradnorm,graddrop",
                ],
            )
            mtl_methods = (
                tuple(mtl_string.split(","))
                if "," in mtl_string
                else tuple([mtl_string])
            )
        else:
            mtl_methods = ("identity",)
        return Hyperparameters(
            dim_s=dim_s,
            dim_y=dim_y,
            dropout=0.1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=opt.max_epochs,
            lstm_layers=lstm_layers,
            mtl_methods=mtl_methods,
            dim_ys=dim_ys,
            dim_z=dim_z,
            early_stopping=opt.early_stopping,
            early_stopping_threshold=opt.early_stopping_threshold,
            patience=opt.patience,
        )

    def get_module_config(
        self, hyperparameters: Hyperparameters
    ) -> Tuple[DataModuleConfig, ...]:
        """
        Get the module configuration based on the hyperparameters.

        Args:
            hyperparameters: The Hyperparameters object.

        Returns:
            A tuple of DataModuleConfig objects.
        """
        module_configs = []
        for module_name in self.dataset.module_names:
            module = self.dataset.get_modules(module_name)[0]
            module_config = DataModuleConfig(
                name=module_name,
                variable_types=module.variable_types,
                n_layers=hyperparameters.lstm_layers,
                num_timepoints=self.dataset.visits_per_module[module_name],
            )
            module_configs.append(module_config)
        return tuple(module_configs)

    def train(self, best_parameters: Hyperparameters) -> GenericMHivaeModel:
        whole_dataloader = self.get_dataloader(
            self.dataset, best_parameters.batch_size, shuffle=True
        )
        module_config = self.get_module_config(best_parameters)
        self.model_config = ModularHivaeConfig(
            module_config=module_config,
            dim_z=best_parameters.dim_z,
            dim_s=best_parameters.dim_s,
            dim_y=best_parameters.dim_y,
            dropout=best_parameters.dropout,
            mtl_method=best_parameters.mtl_methods,
            n_layers=best_parameters.lstm_layers,
            use_imputation_layer=self.config.training.use_imputation_layer,
            dim_ys=best_parameters.dim_ys,
        )
        self.model = self.init_model(self.model_config)
        if isinstance(best_parameters.learning_rate, Dict):
            order = self.dataset.module_names + ["shared"]
            learning_rate = tuple(
                best_parameters.learning_rate[module_name]
                for module_name in order
            )
        else:
            learning_rate = best_parameters.learning_rate
        with mlflow.start_run(run_name=f"{self.run_base}_final_fit"):
            mlflow.log_params(best_parameters.__dict__)
            self.model.fit(
                train_dataloader=whole_dataloader,
                num_epochs=best_parameters.epochs,
                learning_rate=learning_rate,
                early_stopping=False,
            )
        return self.model

    def _objective(self, trial: optuna.Trial) -> float | Tuple[float, float]:
        """
        Objective function for the optimization process.

        Args:
            trial (optuna.Trial): The trial object for hyperparameter optimization.

        Returns:
            float | Tuple[float, float]: The loss value or a tuple of loss values.
        """

        trial_params = self.hyperparameters(trial)
        logger.info(f"Trial parameters: {trial_params}")
        fold_loss = []
        n_epochs = []
        rel_corr_loss = []
        auc_metric = []
        module_config = self.get_module_config(trial_params)
        model_config = ModularHivaeConfig(
            module_config=module_config,
            dim_z=trial_params.dim_z,
            dim_y=trial_params.dim_y,
            dropout=trial_params.dropout,
            mtl_method=trial_params.mtl_methods,
            n_layers=trial_params.lstm_layers,
            use_imputation_layer=self.config.training.use_imputation_layer,
            dim_s=trial_params.dim_s,
            shared_element=self.shared_element,
            dim_ys=trial_params.dim_ys,
        )

        if isinstance(trial_params.learning_rate, Dict):
            order = self.dataset.module_names + ["learning_rate_shared"]
            learning_rate = tuple(
                trial_params.learning_rate[module_name] for module_name in order
            )
        else:
            learning_rate = trial_params.learning_rate
        for i, (train_data, val_data) in enumerate(
            self.cv_generator(self.config.optimization.folds)
        ):
            train_dataloader = self.get_dataloader(
                train_data, trial_params.batch_size, shuffle=True
            )
            val_dataloader = self.get_dataloader(
                val_data, trial_params.batch_size, shuffle=False
            )
            model = self.init_model(model_config)
            with mlflow.start_run(
                run_name=f"{self.run_base}_T{trial._trial_id}_F{i}",
                nested=True,
            ):
                mlflow.log_params(trial_params.__dict__)
                start = time.time()
                try:
                    fit_loss, fit_epoch = model.fit(
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        num_epochs=trial_params.epochs,
                        learning_rate=learning_rate,
                        early_stopping=trial_params.early_stopping,
                        early_stopping_threshold=trial_params.early_stopping_threshold,
                        patience=trial_params.patience,
                    )
                except ValueError:
                    raise optuna.TrialPruned(
                        "Trial pruned due to error during training. Likely due to unsuitable hyperparameters"
                    )

                if ((time.time() - start) / 3600) > 2:
                    logger.warning(
                        f"Trial pruned due to very long execution ({(time.time() - start) / 3600}h at fold {i})"
                    )
                    raise optuna.TrialPruned()

                res = model.predict(val_dataloader)
                mlflow.log_metric("Final loss", res.avg_loss)
                logger.info(f"Fold {i} loss: {res.avg_loss}")

                fold_loss.append(res.avg_loss)
                n_epochs.append(fit_epoch)

                if (
                    self.config.optimization.use_relative_correlation_error_for_optimization
                    or self.config.optimization.use_auc_for_optimization
                ):
                    # Calculate the relative correlation loss
                    assert len(res.outputs) == len(model.module_configs)

                    orig_data = val_data.to_pandas()
                    module_dfs = []
                    for module_output, conf in zip(
                        res.outputs, model.module_configs
                    ):
                        module_columns = self.dataset.get_modules(conf.name)[
                            0
                        ].columns
                        cleaned_columns = [
                            re.sub(r"_VIS[0-9]+", "", c) for c in module_columns
                        ]
                        module_samples = module_output.samples
                        if module_samples.ndim == 2:
                            module_samples = module_samples.unsqueeze(1)

                        module_data = pd.concat(
                            [
                                pd.DataFrame(
                                    module_samples[:, i, :].numpy(),
                                    columns=cleaned_columns,
                                )
                                for i in range(module_samples.shape[1])
                            ]
                        )
                        module_data["SUBJID"] = (
                            val_data.subj * conf.num_timepoints
                        )
                        module_data["VISIT"] = np.repeat(
                            np.arange(1, module_samples.shape[1] + 1),
                            len(val_data),
                        )
                        module_dfs.append(module_data)

                    decoded_data = reduce(
                        lambda x, y: pd.merge(
                            x, y, on=["SUBJID", "VISIT"], how="outer"
                        ),
                        module_dfs,
                    )
                    decoded_data_filtered = decoded_data.loc[
                        :, orig_data.columns
                    ].drop(columns=["SUBJID", "VISIT"])
                    orig_data_filtered = orig_data.drop(
                        columns=["SUBJID", "VISIT"]
                    )

                    fold_rel_corr_loss, m1, m2 = RelativeCorrelation.error(
                        real=orig_data_filtered,
                        synthetic=decoded_data_filtered,
                    )
                    mlflow.log_metric(
                        "Relative correlation loss", fold_rel_corr_loss
                    )
                    rel_corr_loss.append(fold_rel_corr_loss)

                    auc = get_auc(orig_data_filtered, decoded_data_filtered)
                    if auc < 0.5:
                        auc = 1 - auc
                    auc_quality = max(math.floor((1 - auc) * 200), 1)
                    mlflow.log_metric("AUC", auc)
                    mlflow.log_metric("AUC quality", auc_quality)
                    auc_metric.append(auc_quality)

        loss = np.mean(fold_loss)
        best_epoch = n_epochs[np.argmin(fold_loss)]
        mlflow.log_metric("Best epoch", best_epoch)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("n_epochs", n_epochs)

        if self.config.optimization.use_relative_correlation_error_for_optimization:
            rel_corr_loss = np.mean(rel_corr_loss)
            logger.info(f"Trial loss: {loss}; rel_corr_loss: {rel_corr_loss}")
            return loss, rel_corr_loss
        elif self.config.optimization.use_auc_for_optimization:
            return np.mean(auc_metric)
        else:
            logger.info(f"Trial loss: {loss}")
            return loss

    def predict(self, dl: DataLoader) -> ModularHivaeOutput:
        """
        Predict the output of the model.

        Args:
            dl (DataLoader): The DataLoader object.

        Returns:
            ModularHivaeOutput: The output of the model.
        """
        with torch.no_grad():
            return self.model.predict(dl)

    def save_model_config(self, path: Path):
        """
        Save the model configuration to a file.

        Args:
            path (Path): The path to save the model configuration.

        Raises:
            Exception: If the model is None.
            Exception: If the model config is None.
        """
        if self.model is None:
            raise Exception("Model should not be none")

        if self.model_config is None:
            raise Exception("Model config should not be none")

        with path.open("wb") as f:
            dill.dump(self.model_config, f)

    def read_model_config(self, path: Path):
        """
        Read the model configuration from a file.

        Args:
            path (Path): The path to read the model configuration from.
        """
        with path.open("rb") as f:
            self.model_config = dill.load(f)

    def decode(
        self,
        encoding: Union[HivaeEncoding, ModularHivaeEncoding],
        use_mode: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Decode the given encoding to obtain the sampled data.

        Args:
            encoding (Union[HivaeEncoding, ModularHivaeEncoding]): The encoding to decode.
            use_mode (bool, optional): Whether to use the mode for decoding. Defaults to True.

        Returns:
            Dict[str, Tensor]: The decoded sampled data, with module names as keys.
        """
        self.model.eval()
        self.model.decoding = use_mode
        with torch.no_grad():
            sampled_data = self.model.decode(encoding)

        sample_dict = {
            module_name: sampled_data[i]
            for i, module_name in enumerate(self.dataset.module_names)
        }
        return sample_dict


class ModularGanTrainer(ModularTrainer[ModularHivaeConfig, ModularHivae]):
    def init_model(self, config: ModularHivaeConfig) -> ModularHivae:
        return GanModularHivae(
            dim_s=config.dim_s,
            dim_ys=config.dim_ys,
            dim_y=config.dim_y,
            dim_z=config.dim_z,
            module_config=config.module_config,
            mtl_method=config.mtl_method,
            shared_element_type=config.shared_element,
            use_imputation_layer=config.use_imputation_layer,
            noise_size=10,
        )
