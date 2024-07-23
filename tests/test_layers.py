import pytest
from pytest import mark
import torch

from vambn.modelling.models.layers import ImputationLayer


@mark.layer
class ImputationLayerTests:
    def test_init(self):
        ImputationLayer(32)

    @pytest.mark.skipif(
        torch.cuda.device_count() < 1, reason="No GPU available"
    )
    def test_move_to_device(self):
        layer = ImputationLayer(32)
        layer.to(torch.device("cuda"))
        assert layer.imputation_matrix.is_cuda
