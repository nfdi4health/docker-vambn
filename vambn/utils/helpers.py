import logging
import random as rnd
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.core.dtypes.common import is_numeric_dtype

logger = logging.getLogger(__name__)


class NaNHandlingStrategy(Enum):
    """Enumeration of strategies for handling NaN values."""

    accept_inbalance = "accept_inbalance"
    sample_random = "sample_random"
    sample_closest = "sample_closest"
    encode_nan = "encode_nan"


class AggregatedMetric:
    """Class to aggregate and compute average of float metrics."""

    def __init__(self) -> None:
        self._values = []

    def __add__(self, new_value: float) -> None:
        """Adds a new value to the metric list.

        Args:
            new_value: The new float value to add.
        """
        self._values.append(new_value)

    def __call__(self) -> float:
        """Computes the average of the aggregated values.

        Returns:
            The average value of the aggregated metrics.
        """
        return sum(self._values) / len(self._values)


class AggregatedTorchMetric:
    """Class to aggregate and compute average of torch.Tensor metrics."""

    def __init__(self) -> None:
        self._values = []

    def __add__(self, new_value: torch.Tensor) -> "AggregatedTorchMetric":
        """Adds a new tensor value to the metric list.

        Args:
            new_value: The new tensor value to add.

        Returns:
            self: The updated AggregatedTorchMetric object.
        """
        self._values.append(new_value)
        return self

    def __call__(self) -> torch.Tensor:
        """Computes the average of the aggregated tensor values.

        Returns:
            The average tensor value of the aggregated metrics.
        """
        return torch.mean(torch.stack(self._values))


def delete_directory(dir_path: Path) -> None:
    """Deletes a directory and all its contents.

    Args:
        dir_path: Path to the directory to delete.
    """
    dir_path = Path(dir_path)
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_dir():
                # Recursively delete subdirectories
                for subitem in item.rglob("*"):
                    if subitem.is_file():
                        subitem.unlink()
                item.rmdir()
            else:
                # Delete files
                item.unlink()
        dir_path.rmdir()
    else:
        logger.warning("Directory does not exist")


def get_normalized_vector_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes the normalized distance between two vectors.

    Args:
        vec1: The first vector.
        vec2: The second vector.

    Returns:
        The normalized distance between the two vectors.
    """
    diff = np.abs(vec1 - vec2)
    max = np.maximum.reduce([vec1, vec2])
    min = np.minimum.reduce([vec1, vec2])
    range = np.abs(max) + np.abs(min)
    quotient = np.divide(diff, range)
    quotient[np.isnan(quotient)] = 0
    sum = np.sum(quotient)
    return sum


def get_vector_to_mixed_matrix_distance(
    vec: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Computes the distance from a vector to a mixed-type matrix.

    Args:
        vec: The vector to compare.
        matrix: The matrix to compare against.

    Returns:
        An array of distances.
    """
    # get mask for categorical columns
    cat_cols = [column_is_categorical(col) for col in matrix.T]
    # number of different categorical columns for each row of df
    cat_distances = np.sum(matrix[:, cat_cols] == vec[cat_cols], axis=1)
    num_distances = np.array(
        [get_normalized_vector_distance(vec, row) for row in matrix]
    )
    return cat_distances + num_distances


def encode_numerical_columns(patient_data: pd.DataFrame) -> pd.DataFrame:
    """Encodes non-numeric columns of a DataFrame with categorical values.

    Args:
        patient_data: The DataFrame with patient data.

    Returns:
        A DataFrame with non-numeric columns encoded as categorical values.
    """
    tmp = patient_data.copy()
    for column in tmp:
        if not is_numeric_dtype(tmp[column]):
            tmp[column] = tmp[column].astype("category")
            tmp[column] = tmp[column].cat.codes
    return tmp


def column_is_categorical(col: pd.Series) -> bool:
    """Determines if a column is categorical.

    Args:
        col: The column to check.

    Returns:
        True if the column is categorical, False otherwise.
    """
    if col.dtype == "object":
        return True
    # ugly heuristic for encoded categorical values
    elif np.sum(np.asarray(col)) % 1 == 0 and np.max(np.asarray(col)) < 10:
        return True
    else:
        return False


def handle_nan_values(
    real: pd.DataFrame,
    virtual: pd.DataFrame,
    strategy: NaNHandlingStrategy = NaNHandlingStrategy.sample_random,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Handles NaN values in two dataframes according to the specified strategy.

    Args:
        real: The real dataframe.
        virtual: The virtual dataframe.
        strategy: The strategy to use for handling NaN values.

    Returns:
        A tuple containing the processed real and virtual dataframes.
    """
    np.random.seed(42)

    # convert real and virtual to pandas if they are numpy arrays
    if isinstance(real, np.ndarray) or isinstance(real, list):
        real = pd.DataFrame(real)

    if isinstance(virtual, np.ndarray) or isinstance(virtual, list):
        virtual = pd.DataFrame(virtual)

    if not (real.isnull().values.any() or virtual.isnull().values.any()):
        return real, virtual

    if strategy == NaNHandlingStrategy.accept_inbalance:
        return real.dropna(), virtual.dropna()

    if strategy == NaNHandlingStrategy.sample_random:
        # Drop all rows with high NaN ratio
        real = real.dropna(axis=0, thresh=0.8 * real.shape[1])
        virtual = virtual.dropna(axis=0, thresh=0.8 * virtual.shape[1])

        # Drop columns with high NaN ratio
        real = real.dropna(axis=1, thresh=0.8 * real.shape[0])
        # use the same columns for real and virtual
        virtual = virtual[real.columns]

        # remove all rows containing NaN values
        real = real.dropna()
        virtual = virtual.dropna()
        # subsample in such a way that each dataframe has the same amount of rows
        if real.shape[0] < virtual.shape[0]:
            virtual = virtual.loc[
                np.random.choice(virtual.index, real.shape[0], replace=False)
            ]
        elif virtual.shape[0] < real.shape[0]:
            real = real.loc[
                np.random.choice(real.index, virtual.shape[0], replace=False)
            ]

    elif strategy == NaNHandlingStrategy.sample_closest:
        # remove all rows containing NaN values
        real = real.dropna()
        virtual = virtual.dropna()
        # order columns such that identical align in order
        real = real.reindex(sorted(real.columns), axis=1)
        virtual = virtual.reindex(sorted(virtual.columns), axis=1)
        # sample in the bigger set (either real or virtual) the data points that are most similar
        sample_idx = []
        if real.shape[0] < virtual.shape[0]:
            for a in real.to_numpy():
                distance = get_vector_to_mixed_matrix_distance(
                    a, virtual.to_numpy()
                )
                sample_idx.append(np.argmin(distance))
            virtual = virtual.iloc[sample_idx]
        elif virtual.shape[0] < real.shape[0]:
            for a in virtual.to_numpy():
                distance = get_vector_to_mixed_matrix_distance(
                    a, real.to_numpy()
                )
                sample_idx.append(np.argmin(distance))
            real = real.iloc[sample_idx]

    elif strategy == NaNHandlingStrategy.encode_nan:
        for col in real:
            sum_real_na = real[col].isna().sum()
            sum_virtual_na = virtual[col].isna().sum()
            sum_diff = abs(sum_virtual_na - sum_real_na)
            if sum_real_na > sum_virtual_na:
                # sample indices to replace with NaN
                replace_idx = rnd.sample(range(1, virtual.shape[0]), sum_diff)
                virtual.loc[replace_idx, col] = None
            elif sum_real_na < sum_virtual_na:
                # sample indices to replace with NaN
                replace_idx = rnd.sample(range(1, real.shape[0]), sum_diff)
                real.loc[replace_idx, col] = None
            # encode NaN
            cat_col_bool = [
                column_is_categorical(col) for col in real.dropna().to_numpy().T
            ]
            cat_cols = np.array(real.columns)[cat_col_bool]
            num_cols = np.array(real.columns)[np.invert(cat_col_bool)]
            for col in cat_cols:
                real[col] = real[col].fillna(real[col].max() + 1)
                virtual[col] = virtual[col].fillna(virtual[col].max() + 1)
            for col in num_cols:
                real[col] = real[col].fillna(0)
                virtual[col] = virtual[col].fillna(0)

    return real, virtual
