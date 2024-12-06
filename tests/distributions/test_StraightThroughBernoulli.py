import torch
import sys, os

from relaxit.distributions import StraightThroughBernoulli

# Testing reparameterized sampling and log prob from the StraightThroughBernoulli distribution


def test_sample_shape():
    logits = torch.tensor([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    distribution = StraightThroughBernoulli(logits=logits)
    sample = distribution.rsample()
    assert sample.shape == logits.shape


def test_sample_grad():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    distribution = StraightThroughBernoulli(logits=logits)
    sample = distribution.rsample()
    assert sample.requires_grad == True


def test_log_prob_shape():
    logits = torch.tensor([1.0, 2.0, 3.0])
    distribution = StraightThroughBernoulli(logits=logits)
    value = torch.tensor([1.0, 1.0, 1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.shape == torch.Size([3])
    

def test_log_prob_grad():
    logits = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    distribution = StraightThroughBernoulli(logits=logits)
    value = torch.tensor([1.0, 1.0, 1.0])
    log_prob = distribution.log_prob(value)
    assert log_prob.requires_grad == True
