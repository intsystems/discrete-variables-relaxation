import torch
import sys, os

from relaxit.distributions import StochasticTimesSmooth

# Testing reparameterized sampling from the StochasticTimesSmooth distribution


def test_sample_shape():
    logits = torch.tensor([1., 2., 3.])
    distribution = StochasticTimesSmooth(logits=logits)
    samples = distribution.rsample()
    assert samples.shape == torch.Size([3])


def test_sample_grad():
    logits = torch.tensor([1., 2., 3.], requires_grad=True)
    distribution = StochasticTimesSmooth(logits=logits)
    samples = distribution.rsample()
    assert samples.requires_grad == True


def test_log_prob_shape():
    logits = torch.tensor([1., 2., 3.])
    distribution = StochasticTimesSmooth(logits=logits)
    value = torch.Tensor([1., 1., 1.])
    log_prob = distribution.log_prob(value)
    print('log_prob.shape:', log_prob.shape)
    assert log_prob.shape == torch.Size([3])


def test_log_prob_grad():
    logits = torch.tensor([1., 2., 3.], requires_grad=True)
    distribution = StochasticTimesSmooth(logits=logits)
    value = torch.Tensor([1., 1., 1.])
    log_prob = distribution.log_prob(value)
    assert log_prob.requires_grad == True