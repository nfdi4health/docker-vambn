from pytest import mark
import torch
from vambn.modelling.mtl import moo
from vambn.modelling.mtl.parameters import MtlMethodParams


@mark.mtl
class MtlTests:
    def test_forward(self):
        mtl_module = moo.setup_moo([MtlMethodParams("identity")], num_tasks=7)
        moo_block = moo.MooMulti(7, moo_method=mtl_module)

        x_list = [torch.rand(32, 1) for _ in range(7)]
        x_combined = torch.stack(x_list)

        y = moo_block(x_combined)
        i = 0
        for i, y_i in enumerate(y):
            assert y_i.shape == (32, 1)
            assert (y_i == x_list[i]).all()
            i += 1

        assert i == 7
