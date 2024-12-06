import torch
import sys

from relaxit.distributions.LogisticNormalSoftmax import LogisticNormalSoftmax
from relaxit.distributions.approx import (
    lognorm_approximation_fn,
    dirichlet_approximation_fn,
)
from pyro.distributions import Dirichlet

# Testing two-side closed-form Laplace bridge approximation between
# LogisticNormal and Dirichlet distributions


def test_approx():
    # Generate a random concentration parameter
    concentration = torch.randint(1, 10, (3,), dtype=torch.float)

    # Create the Dirichlet distribution
    dirichlet_distribution = Dirichlet(concentration)

    # Approximate the Dirichlet distribution with a LogisticNormal distribution
    lognorm_approximation = lognorm_approximation_fn(dirichlet_distribution)
    loc = lognorm_approximation.loc
    scale = lognorm_approximation.scale

    # Approximate the LogisticNormal distribution with a Dirichlet distribution
    dirichlet_approximation = dirichlet_approximation_fn(lognorm_approximation)
    concentration_approx = dirichlet_approximation.concentration

    # Assert that the original and approximated concentration parameters are close
    assert torch.allclose(concentration, concentration_approx)
