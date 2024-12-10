import torch
from pyro.distributions import Bernoulli


class StraightThroughBernoulli(Bernoulli):
    r"""
    Implementation of the Straight Through Bernoulli from https://arxiv.org/abs/1910.02176.

    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    However, supports gradient flow through parameters due to the
    straight through gradient estimator.

    Args:
        probs (torch.Tensor, optional): Event probabilities.
        logits (torch.Tensor, optional): Event log-odds.
        validate_args (bool, optional): Whether to validate arguments. Defaults to None.
    """

    has_rsample = True

    def __init__(self, *args, **kwargs):
        r"""
        Initializes the StraightThroughBernoulli distribution.

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
        probs = self.probs.expand(shape)
        sample = torch.bernoulli(probs).detach()
        return sample + probs - probs.detach()
