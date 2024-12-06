import torch
import sys

from relaxit.distributions import LogisticNormalSoftmax

# Testing reparameterized sampling from the LogisticNormalSoftmax distribution


def test_sample_shape():
    loc = torch.zeros(3, 4, 5)
    scale = torch.ones(3, 4, 5)
    distribution = LogisticNormalSoftmax(loc, scale)
    sample = distribution.rsample()
    assert sample.shape == torch.Size([3, 4, 5])


def test_sample_grad():
    loc = torch.zeros(3, 4, 5, requires_grad=True)
    scale = torch.ones(3, 4, 5, requires_grad=True)
    distribution = LogisticNormalSoftmax(loc, scale)
    sample = distribution.rsample()
    assert sample.requires_grad == True
