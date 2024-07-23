import torch
from pytest import mark

from vambn.modelling.distributions.gumbel_distribution import GumbelDistribution


@mark.gumbel
class GumbelDistributionTests:

    def test_sample(self):
        distribution = GumbelDistribution(temperature=1e-3, probs=torch.tensor([0.2, 0.3, 0.5]))
        sample = distribution.sample()
        assert sample.shape == torch.Size([3])

    def test_rsample(self):
        distribution = GumbelDistribution(temperature=1e-3, probs=torch.tensor([0.2, 0.3, 0.5]))
        rsample = distribution.rsample()
        assert rsample.shape == torch.Size([3])

    def test_mean(self):
        distribution = GumbelDistribution(temperature=1e-3, probs=torch.tensor([0.2, 0.3, 0.5]))
        mean = distribution.mean
        assert mean.shape == torch.Size([3])

    def test_expand(self):
        distribution = GumbelDistribution(temperature=1e-3, probs=torch.tensor([0.2, 0.3, 0.5]))
        expanded_distribution = distribution.expand(torch.Size([2, 3]))
        assert expanded_distribution.probs.shape == torch.Size([2, 3])

    def test_log_prob(self):
        distribution = GumbelDistribution(temperature=1e-3, probs=torch.tensor([0.2, 0.3, 0.5]))
        value = torch.tensor([0, 1, 0])
        log_prob = distribution.log_prob(value)
        assert log_prob.shape == torch.Size([])

    def test_for_nan(self):
        for _ in range(10000):
            random_logits = torch.randn((1000, 3))
            dist = GumbelDistribution(logits=random_logits, temperature=1.0)
            sample = dist.rsample()
            assert not torch.isnan(sample).any()

