import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
import warnings


class LogStdScaler(BaseEstimator, TransformerMixin):
    """
    A custom scaler that applies a log transformation followed by standard scaling.

    This class is deprecated and will be removed soon.

    Attributes:
        scaler (StandardScaler): The standard scaler used after the log transformation.
    """

    def __init__(self):
        """
        Initializes the LogStdScaler.
        """
        self.scaler = None
        warnings.warn(
            "This class is deprecated and will be removed soon",
            DeprecationWarning,
        )

    def fit(self, X, y=None):
        """
        Fits the scaler to the data after applying a log transformation.

        Args:
            X (array-like): The data to fit.
            y (None, optional): Ignored.

        Returns:
            LogStdScaler: The fitted scaler.
        """
        X_log = np.log1p(X)
        self.scaler = StandardScaler().fit(X_log)
        return self

    def transform(self, X, y=None):
        """
        Transforms the data using the fitted scaler after applying a log transformation.

        Args:
            X (array-like): The data to transform.
            y (None, optional): Ignored.

        Returns:
            array-like: The transformed data.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if self.scaler is None:
            raise RuntimeError(
                "You must fit the scaler before transforming data"
            )
        X_log = np.log1p(X)
        X_scaled = self.scaler.transform(X_log)
        return X_scaled

    def inverse_transform(self, X, y=None):
        """
        Inversely transforms the data using the fitted scaler.

        Args:
            X (array-like): The data to inverse transform.
            y (None, optional): Ignored.

        Returns:
            array-like: The inversely transformed data.
        """
        X_reversed = self.scaler.inverse_transform(X)
        return X_reversed

    def __str__(self) -> str:
        """
        Returns a string representation of the LogStdScaler.

        Returns:
            str: The string representation.
        """
        return "LogStdScaler"
