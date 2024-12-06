import torch
import sys

from relaxit.distributions import InvertibleGaussian
from relaxit.distributions.kl import kl_divergence

# Testing KL-divergence between two IntertibleGaussian distributions


def test_igr_kl_shape():
    loc_1 = torch.zeros(3, 4, 5, requires_grad=True)
    scale_1 = torch.ones(3, 4, 5, requires_grad=True)
    temperature_1 = torch.tensor([1e-0])
    dist_1 = InvertibleGaussian(loc_1, scale_1, temperature_1)

    loc_2 = torch.ones(3, 4, 5, requires_grad=True)  # ones, not zeros
    scale_2 = torch.ones(3, 4, 5, requires_grad=True)
    temperature_2 = torch.tensor([1e-2])
    dist_2 = InvertibleGaussian(loc_2, scale_2, temperature_2)

    div = kl_divergence(dist_1, dist_2)
    assert div.shape == torch.Size([3, 4, 5])


def test_igr_kl_grad():
    loc_1 = torch.zeros(3, 4, 5, requires_grad=True)
    scale_1 = torch.ones(3, 4, 5, requires_grad=True)
    temperature_1 = torch.tensor([1e-0])
    dist_1 = InvertibleGaussian(loc_1, scale_1, temperature_1)

    loc_2 = torch.ones(3, 4, 5, requires_grad=True)  # ones, not zeros
    scale_2 = torch.ones(3, 4, 5, requires_grad=True)
    temperature_2 = torch.tensor([1e-2])
    dist_2 = InvertibleGaussian(loc_2, scale_2, temperature_2)

    div = kl_divergence(dist_1, dist_2)
    assert div.requires_grad == True


def test_igr_kl_value():
    loc_1 = torch.ones(3, 4, 5, requires_grad=True)
    scale_1 = torch.ones(3, 4, 5, requires_grad=True)
    temperature_1 = torch.tensor([1e-2])
    dist_1 = InvertibleGaussian(loc_1, scale_1, temperature_1)

    loc_2 = torch.ones(3, 4, 5, requires_grad=True)  # ones, not zeros
    scale_2 = torch.ones(3, 4, 5, requires_grad=True)
    temperature_2 = torch.tensor([1e-2])
    dist_2 = InvertibleGaussian(loc_2, scale_2, temperature_2)

    div = kl_divergence(dist_1, dist_2)
    assert torch.allclose(div, torch.zeros_like(div))
