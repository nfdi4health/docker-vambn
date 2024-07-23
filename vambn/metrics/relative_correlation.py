import logging
from typing import Any

import numpy as np
import pandas as pd

from vambn import HIVAE_EPS

logger = logging.getLogger(__name__)


class RelativeCorrelation:
    """Class for calculating relative correlation metrics between data sets."""

    @staticmethod
    def error(
        real: pd.DataFrame, synthetic: pd.DataFrame, method: str = "spearman"
    ) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
        """
        Calculate the relative error of correlation between two pandas DataFrames.

        Args:
            real (pd.DataFrame): First DataFrame.
            synthetic (pd.DataFrame): Second DataFrame.
            method (str): Method for correlation. Defaults to "spearman".

        Returns:
            tuple: A tuple containing:
                - float: The relative error of correlation between the two DataFrames.
                - pd.DataFrame: The correlation matrix of the real DataFrame.
                - pd.DataFrame: The correlation matrix of the synthetic DataFrame.
        """
        # calculate the correlation matrices
        # ensure both dataframes have float64 dtypes
        real = real.astype("float64")
        synthetic = synthetic.astype("float64")

        corr_real = real.corr(method=method)
        corr_synthetic = synthetic.corr(method=method)

        # calculate the difference between the correlation matrices
        diff_corr = corr_synthetic.values - corr_real.values

        # calculate the relative error
        rel_error = np.linalg.norm(diff_corr) / (
            np.linalg.norm(corr_real.values) + HIVAE_EPS
        )

        return rel_error, corr_real, corr_synthetic
