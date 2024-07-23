import pytest
from pytest import mark
import torch
from vambn.modelling.distributions.truncated_normal import TruncatedNormal


@mark.distributions
class TruncatedNormalTests:
    def test_positive_samples(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        low = torch.tensor(0.0)
        high = torch.tensor(float("inf"))
        dist = TruncatedNormal(loc, scale, low, high)
        samples = dist.sample((1000,))
        assert (samples >= low).all()

    def test_upper_bound_samples(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        low = torch.tensor(-float("inf"))
        high = torch.tensor(1.0)
        dist = TruncatedNormal(loc, scale, low, high)
        samples = dist.sample((1000,))
        assert (samples <= high).all()

    def test_lower_and_upper_bound_samples(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        low = torch.tensor(-1.0)
        high = torch.tensor(1.0)
        dist = TruncatedNormal(loc, scale, low, high)
        samples = dist.sample((1000,))
        assert (samples >= low).all() and (samples <= high).all()

    def test_log_prob(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        low = torch.tensor(-1.0)
        high = torch.tensor(1.0)
        dist = TruncatedNormal(loc, scale, low, high)
        value = torch.tensor(0.5)
        log_prob = dist.log_prob(value)
        assert log_prob.shape == ()
        assert not torch.isinf(log_prob)

    def test_log_prob_outside_bounds(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        low = torch.tensor(-1.0)
        high = torch.tensor(1.0)
        dist = TruncatedNormal(loc, scale, low, high)
        value = torch.tensor(2.0)
        log_prob = dist.log_prob(value)
        assert log_prob.shape == ()
        assert log_prob < 100

    def test_cdf(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        low = torch.tensor(-1.0)
        high = torch.tensor(1.0)
        dist = TruncatedNormal(loc, scale, low, high)
        value = torch.tensor(0.5)
        cdf = dist.cdf(value)
        assert cdf.shape == ()
        assert 0 <= cdf <= 1

    def test_icdf(self):
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        low = torch.tensor(-1.0)
        high = torch.tensor(1.0)
        dist = TruncatedNormal(loc, scale, low, high)
        value = torch.tensor(0.5)
        icdf = dist.icdf(value)
        assert icdf.shape == ()
        assert low <= icdf <= high

    def test_lower_constraint(self):
        loc = torch.tensor(10.0)
        scale = torch.tensor(3)
        low = torch.tensor(0)
        dist = TruncatedNormal(loc, scale, low=low)
        samples = dist.sample((1000,))
        assert torch.all(samples >= low)
