import torch
import sys
from pyro.distributions import Dirichlet

from relaxit.distributions.LogisticNormalSoftmax import LogisticNormalSoftmax
from relaxit.distributions.approx import (
    lognorm_approximation_fn,
    dirichlet_approximation_fn,
)


def test_approx():
    """
    This function tests the two-side closed-form Laplace bridge approximation
    between LogisticNormal and Dirichlet distributions.

    Steps:
    1. Generate a random concentration parameter for the Dirichlet distribution.
    2. Create a Dirichlet distribution using the generated concentration parameter.
    3. Approximate the Dirichlet distribution with a LogisticNormal distribution.
    4. Extract the location (loc) and scale parameters from the LogisticNormal approximation.
    5. Approximate the LogisticNormal distribution back to a Dirichlet distribution.
    6. Extract the concentration parameter from the Dirichlet approximation.
    7. Assert that the original and approximated concentration parameters are close.
    """

    # Step 1: Generate a random concentration parameter
    # The concentration parameter is generated using random integers between 1 and 10
    # with a shape of (3,) and a data type of float.
    concentration = torch.randint(1, 10, (3,), dtype=torch.float)

    # Step 2: Create the Dirichlet distribution
    # The Dirichlet distribution is created using the generated concentration parameter.
    dirichlet_distribution = Dirichlet(concentration)

    # Step 3: Approximate the Dirichlet distribution with a LogisticNormal distribution
    # The lognorm_approximation_fn function is used to approximate the Dirichlet distribution
    # with a LogisticNormal distribution.
    lognorm_approximation = lognorm_approximation_fn(dirichlet_distribution)

    # Step 4: Extract the location (loc) and scale parameters from the LogisticNormal approximation
    # The location and scale parameters are extracted from the LogisticNormal approximation.
    loc = lognorm_approximation.loc
    scale = lognorm_approximation.scale

    # Step 5: Approximate the LogisticNormal distribution back to a Dirichlet distribution
    # The dirichlet_approximation_fn function is used to approximate the LogisticNormal distribution
    # back to a Dirichlet distribution.
    dirichlet_approximation = dirichlet_approximation_fn(lognorm_approximation)

    # Step 6: Extract the concentration parameter from the Dirichlet approximation
    # The concentration parameter is extracted from the Dirichlet approximation.
    concentration_approx = dirichlet_approximation.concentration

    # Step 7: Assert that the original and approximated concentration parameters are close
    # The assert statement checks that the original and approximated concentration parameters
    # are close using the torch.allclose function.
    assert torch.allclose(concentration, concentration_approx)
