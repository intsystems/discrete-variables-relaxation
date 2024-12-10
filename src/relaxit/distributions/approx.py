import torch
from .LogisticNormalSoftmax import LogisticNormalSoftmax
from pyro.distributions import Dirichlet


def lognorm_approximation_fn(
    dirichlet_distribution: Dirichlet,
) -> LogisticNormalSoftmax:
    r"""
    Approximates a Dirichlet distribution with a LogisticNormalSoftmax distribution.

    Args:
        dirichlet_distribution (Dirichlet): The Dirichlet distribution to approximate.

    Returns:
        LogisticNormalSoftmax: The approximated LogisticNormalSoftmax distribution.
    """
    concentration = dirichlet_distribution.concentration
    num_events = torch.tensor(dirichlet_distribution.event_shape, dtype=torch.float)

    # Compute the location parameter (mu)
    loc = concentration.log() - (1 / num_events) * concentration.log().sum(
        -1
    ).unsqueeze(-1)

    # Compute the scale parameter (sigma)
    scale = 1 / concentration - (1 / num_events) * (
        2 / concentration - (1 / num_events) * (1 / concentration).sum(-1).unsqueeze(-1)
    )

    # Create the LogisticNormalSoftmax distribution
    lognorm_approximation = LogisticNormalSoftmax(loc, scale)

    return lognorm_approximation


def dirichlet_approximation_fn(
    lognorm_distribution: LogisticNormalSoftmax,
) -> Dirichlet:
    r"""
    Approximates a LogisticNormalSoftmax distribution with a Dirichlet distribution.

    Args:
        lognorm_distribution (LogisticNormalSoftmax): The LogisticNormalSoftmax distribution to approximate.

    Returns:
        Dirichlet: The approximated Dirichlet distribution.
    """
    num_events = torch.tensor(lognorm_distribution.event_shape, dtype=torch.float)
    loc, scale = lognorm_distribution.loc, lognorm_distribution.scale

    # Compute the concentration parameter (alpha)
    concentration = (1 / scale) * (
        1
        - 2 / num_events
        + loc.exp() / (num_events**2) * loc.neg().exp().sum(-1).unsqueeze(-1)
    )

    # Create the Dirichlet distribution
    dirichlet_approximation = Dirichlet(concentration)

    return dirichlet_approximation
