import numpy as np
import pytest
import torch

from vambn.metrics import jensen_shannon_distance


@pytest.mark.metric
class JensenShannonDistanceTests:
    def test_real_data(self):
        tensor1 = np.random.normal(0, 1, 1000)
        tensor2 = np.random.normal(0.5, 1, 1000)
        distance = jensen_shannon_distance(tensor1, tensor2, "real")
        assert isinstance(distance, float)

    def test_pos_data(self):
        tensor1 = np.random.lognormal(0, 1, 1000)
        tensor2 = np.random.lognormal(0.5, 1, 1000)
        distance = jensen_shannon_distance(tensor1, tensor2, "pos")
        assert isinstance(distance, float)

    def test_count_data(self):
        tensor1 = np.random.poisson(5, 1000)
        tensor2 = np.random.poisson(7, 1000)
        distance = jensen_shannon_distance(tensor1, tensor2, "count")
        assert isinstance(distance, float)

    def test_cat_data(self):
        tensor1 = np.random.choice(['a', 'b', 'c'], 1000)
        tensor2 = np.random.choice(['a', 'b', 'c'], 1000)
        distance = jensen_shannon_distance(tensor1, tensor2, "cat")
        assert isinstance(distance, float)

    def test_tensor_input(self):
        tensor1 = torch.tensor(np.random.normal(0, 1, 1000))
        tensor2 = torch.tensor(np.random.normal(0.5, 1, 1000))
        distance = jensen_shannon_distance(tensor1, tensor2, "real")
        assert isinstance(distance, float)

    def test_equal_tensors(self):
        tensor1 = np.random.normal(0, 1, 1000)
        tensor2 = tensor1
        distance = jensen_shannon_distance(tensor1, tensor2, "real")
        assert distance == 0

    def test_different_tensors(self):
        tensor1 = np.random.normal(0, 1, 1000)
        tensor2 = np.random.normal(25, 10, 1000)
        distance = jensen_shannon_distance(tensor1, tensor2, "real")
        assert distance > 0.8 and distance < 1.0, f"Distance was {distance}"

    def test_different_categories(self):
        tensor1 = np.random.choice([0, 1, 2], 1000)
        tensor2 = np.random.choice([3,4,5], 1000)
        distance = jensen_shannon_distance(tensor1, tensor2, "cat")
        assert distance > 0.8 and distance < 1.0, f"Distance was {distance}"

    def test_unknown_data_type(self):
        tensor1 = np.random.normal(0, 1, 1000)
        tensor2 = np.random.normal(0.5, 1, 1000)
        with pytest.raises(Exception):
            jensen_shannon_distance(tensor1, tensor2, "unknown")
