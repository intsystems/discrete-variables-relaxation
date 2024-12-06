import torch
import sys, os

from relaxit.distributions import HardConcrete

# Testing reparameterized sampling from the HardConcrete distribution


def test_sample_shape():
    alpha = torch.tensor([1.0])
    beta = torch.tensor([2.0])
    gamma = torch.tensor([-3.0])
    xi = torch.tensor([4.0])
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)
    samples = distribution.rsample(sample_shape=torch.Size([3]))
    assert samples.shape == torch.Size([3, 1])


def test_sample_grad():
    alpha = torch.tensor([1.0], requires_grad=True)
    beta = torch.tensor([2.0], requires_grad=True)
    gamma = torch.tensor([-3.0], requires_grad=True)
    xi = torch.tensor([4.0], requires_grad=True)
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)
    samples = distribution.rsample(sample_shape=torch.Size([3]))
    assert samples.requires_grad == True


def test_log_prob_shape():
    alpha = torch.tensor([1.0])
    beta = torch.tensor([2.0])
    gamma = torch.tensor([-3.0])
    xi = torch.tensor([4.0])
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)
    value = torch.tensor([1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.shape == torch.Size([1])


def test_log_prob_grad():
    alpha = torch.tensor([1.0], requires_grad=True)
    beta = torch.tensor([2.0], requires_grad=True)
    gamma = torch.tensor([-3.0], requires_grad=True)
    xi = torch.tensor([4.0], requires_grad=True)
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)
    value = torch.tensor([1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.requires_grad == True