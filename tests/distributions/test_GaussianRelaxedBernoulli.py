import torch
import sys, os

from relaxit.distributions import GaussianRelaxedBernoulli

# Testing reparameterized sampling from the GaussianRelaxedBernoulli distribution


def test_sample_shape():
    loc = torch.tensor([0.0])
    scale = torch.tensor([1.0])
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)
    samples = distribution.rsample(sample_shape=torch.Size([3]))
    assert samples.shape == torch.Size([3, 1])


def test_sample_grad():
    loc = torch.tensor([0.0], requires_grad=True)
    scale = torch.tensor([1.0], requires_grad=True)
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)
    samples = distribution.rsample()
    assert samples.requires_grad == True


def test_log_prob_shape():
    loc = torch.tensor([0.0])
    scale = torch.tensor([1.0])
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)
    value = torch.tensor([1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.shape == torch.Size([1])


def test_log_prob_grad():
    loc = torch.tensor([0.0], requires_grad=True)
    scale = torch.tensor([1.0], requires_grad=True)
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)
    value = torch.tensor([1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.requires_grad == True