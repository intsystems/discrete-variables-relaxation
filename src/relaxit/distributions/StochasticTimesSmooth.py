import torch
from pyro.distributions import Bernoulli


class StochasticTimesSmooth(Bernoulli):
    r"""
    Implementation of the Stochastic Times Smooth from https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=62c76ca0b2790c34e85ba1cce09d47be317c7235.

    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    However, supports gradient flow through parameters due to the
    stochastic times smooth gradient estimator.

    Args:
        probs (torch.Tensor, optional): Event probabilities.
        logits (torch.Tensor, optional): Event log-odds.
        validate_args (bool, optional): Whether to validate arguments. Defaults to None.
    """

    has_rsample = True

    def __init__(self, *args, **kwargs):
        r"""
        Initializes the StochasticTimesSmooth distribution.

        Args:
            *args: Positional arguments for the Bernoulli distribution.
            **kwargs: Keyword arguments for the Bernoulli distribution.
        """
        super().__init__(*args, **kwargs)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution using the reparameterization trick.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: A sample from the distribution.
        """
        shape = self._extended_shape(sample_shape)
        sqrt_probs = self.probs.expand(shape).sqrt()
        sample = sqrt_probs * torch.bernoulli(sqrt_probs)
        return sample