import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from torch import Tensor

from vambn.utils.helpers import handle_nan_values


def jensen_shannon_distance_kde(
    tensor1: np.ndarray | Tensor,
    tensor2: np.ndarray | Tensor,
    data_type: str,
    bins: int = 30,
) -> float:
    """
    Calculate the Jensen-Shannon distance between two tensors.

    Args:
        tensor1 (np.ndarray | Tensor): Tensor 1.
        tensor2 (np.ndarray | Tensor): Tensor 2.
        data_type (str): Type of data. Possible values are "real", "pos", "truncate_norm", "count", "cat", "truncate_norm", "gamma".
        bins (int, optional): Number of bins for count. Defaults to 30.

    Returns:
        float: Jensen-Shannon distance.

    Raises:
        Exception: If the data type is unknown.
    """
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.detach().cpu().numpy()
    if torch.is_tensor(tensor2):
        tensor2 = tensor2.detach().cpu().numpy()

    if data_type in ["real", "pos", "truncate_norm", "gamma"]:
        kde1 = gaussian_kde(tensor1)
        kde2 = gaussian_kde(tensor2)

        # Evaluate the KDEs on a common set of points
        min_val = min(tensor1.min(), tensor2.min())
        max_val = max(tensor1.max(), tensor2.max())
        points = np.linspace(min_val, max_val, 1000)
        pdf1 = kde1(points)
        pdf2 = kde2(points)
    elif data_type == "count":
        max_val = max(tensor1.max(), tensor2.max())
        bins = (
            np.arange(0, max_val + 2) - 0.5
        )  # Bins edges are halfway between integers
        pdf1, _ = np.histogram(tensor1, bins=bins, density=True)
        pdf2, _ = np.histogram(tensor2, bins=bins, density=True)
    elif data_type == "cat" or data_type == "categorical":
        # Calculate probability distribution based on the frequency of each category
        categories = np.union1d(tensor1, tensor2)
        pdf1 = np.array(
            [np.mean(tensor1 == category) for category in categories]
        )
        pdf2 = np.array(
            [np.mean(tensor2 == category) for category in categories]
        )
    else:
        raise Exception(f"Unknown data type {data_type}")

    # Calculate the Jensen-Shannon distance
    return jensenshannon(pdf1, pdf2)


def jensen_shannon_distance(
    real: np.ndarray | Tensor,
    synthetic: np.ndarray | Tensor,
    data_type: str,
) -> float:
    """
    Calculate the Jensen-Shannon distance between two tensors.

    Args:
        real (np.ndarray | Tensor): Real data tensor.
        synthetic (np.ndarray | Tensor): Synthetic data tensor.
        data_type (str): Type of data. Possible values are "real", "pos", "truncate_norm", "count", "cat", "truncate_norm".

    Returns:
        float: Jensen-Shannon distance.

    Raises:
        Exception: If the data type is unknown or all columns contain too many NaN values and were removed.
    """
    if torch.is_tensor(real):
        real = real.detach().cpu().numpy()
    if torch.is_tensor(synthetic):
        synthetic = synthetic.detach().cpu().numpy()

    real, synthetic = handle_nan_values(real, synthetic)
    if real.shape[1] == 0:
        raise Exception(
            "All columns contain too many NaN values and were removed."
        )
    real = real.iloc[:, 0].to_numpy()
    synthetic = synthetic.iloc[:, 0].to_numpy()

    if data_type in ["real", "pos", "truncate_norm", "count", "gamma"]:
        try:
            n_bins = np.histogram_bin_edges(real, bins="auto")
        except:  # noqa
            n_bins = np.histogram_bin_edges(real, bins=50)
        if len(n_bins) > 1000:
            n_bins = np.histogram_bin_edges(real, bins=100)
        real_binned = np.bincount(np.digitize(real, n_bins))
        synthetic_binned = np.bincount(np.digitize(synthetic, n_bins))
    elif data_type == "cat" or data_type == "categorical":
        # Calculate probability distribution based on the frequency of each category
        if (
            synthetic.dtype == np.float64
            or synthetic.dtype == np.float32
            or synthetic.dtype == np.float16
        ):
            synthetic = synthetic.astype(int)

        categories = np.union1d(real, synthetic)
        real_binned = np.array(
            [np.sum(real == category) for category in categories]
        )
        synthetic_binned = np.array(
            [np.sum(synthetic == category) for category in categories]
        )
    else:
        raise Exception(f"Unknown data type {data_type}")

    if len(real_binned) != len(synthetic_binned):
        padding_length = np.abs(len(real_binned) - len(synthetic_binned))
        if len(real_binned) > len(synthetic_binned):
            synthetic_binned = np.pad(synthetic_binned, (0, padding_length))
        else:
            real_binned = np.pad(real_binned, (0, padding_length))

    # Calculate the Jensen-Shannon distance
    return jensenshannon(real_binned, synthetic_binned)
