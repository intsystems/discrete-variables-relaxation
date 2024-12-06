import torch
import sys

from relaxit.distributions import InvertibleGaussian

# Testing reparameterized sampling from the InvertibleGaussian distribution


def test_sample_shape():
    loc = torch.zeros(3, 4, 5)
    scale = torch.ones(3, 4, 5)
    temperature = torch.tensor([1e-0])
    distribution = InvertibleGaussian(loc, scale, temperature)
    sample = distribution.rsample()
    assert sample.shape == torch.Size([3, 4, 6])


def test_sample_grad():
    loc = torch.zeros(3, 4, 5, requires_grad=True)
    scale = torch.ones(3, 4, 5, requires_grad=True)
    temperature = torch.tensor([1e-0])
    distribution = InvertibleGaussian(loc, scale, temperature)
    sample = distribution.rsample()
    assert sample.requires_grad == True


def test_log_prob_shape():
    loc = torch.zeros(3, 4, 5)
    scale = torch.ones(3, 4, 5)
    temperature = torch.tensor([1e-0])
    distribution = InvertibleGaussian(loc, scale, temperature)
    value = 0.5 * torch.ones(3, 4, 6)
    log_prob = distribution.log_prob(value)
    assert log_prob.shape == torch.Size([3, 4, 5])


def test_log_prob_grad():
    loc = torch.zeros(3, 4, 5, requires_grad=True)
    scale = torch.ones(3, 4, 5, requires_grad=True)
    temperature = torch.tensor([1e-0])
    distribution = InvertibleGaussian(loc, scale, temperature)
    value = 0.5 * torch.ones(3, 4, 6)
    log_prob = distribution.log_prob(value)
    assert log_prob.requires_grad == True
