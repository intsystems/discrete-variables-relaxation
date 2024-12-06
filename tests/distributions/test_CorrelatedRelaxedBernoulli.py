import torch
import sys, os

from relaxit.distributions import CorrelatedRelaxedBernoulli

# Testing reparameterized sampling from the GaussianRelaxedBernoulli distribution


def test_sample_shape():
    pi = torch.tensor([0.1, 0.2, 0.3])
    R = torch.tensor([[1.0]])
    tau = torch.tensor([2.0])
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)
    samples = distribution.rsample()
    assert samples.shape == torch.Size([3])


def test_sample_grad():
    pi = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    R = torch.tensor([[1.0]])
    tau = torch.tensor([2.0])
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)
    samples = distribution.rsample()
    assert samples.requires_grad == True


def test_log_prob_shape():
    pi = torch.tensor([0.1, 0.2, 0.3])
    R = torch.tensor([[1.0]])
    tau = torch.tensor([2.0])
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)
    value = torch.tensor([1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.shape == torch.Size([3])


def test_log_prob_grad():
    pi = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    R = torch.tensor([[1.0]])
    tau = torch.tensor([2.0])
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)
    value = torch.tensor([1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.requires_grad == True