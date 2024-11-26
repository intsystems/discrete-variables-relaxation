#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The :mod:`relaxit.distributions.CorrelatedRelaxedBernoulli` contains classes:

- :class:`relaxit.distributions.CorrelatedRelaxedBernoulli.CorrelatedRelaxedBernoulli`

"""
from __future__ import print_function

__docformat__ = "restructuredtext"

import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.normal import Normal


class CorrelatedRelaxedBernoulli(TorchDistribution):
    r"""
    Correlated Relaxed Bernoulli distribution class inheriting from Pyro's TorchDistribution.

    :param pi: Selection probability vector.
    :type pi: torch.Tensor
    :param R: Covariance matrix.
    :type R: torch.Tensor
    :param tau: Temperature hyper-parameter.
    :type tau: torch.Tensor
    """

    arg_constraints = {
        "pi": constraints.interval(0, 1),
        "R": constraints.positive_definite,
        "tau": constraints.positive,
    }
    support = constraints.interval(0, 1)
    has_rsample = True

    def __init__(
        self,
        pi: torch.Tensor,
        R: torch.Tensor,
        tau: torch.Tensor,
        validate_args: bool = None,
    ):
        r"""Initializes the CorrelatedRelaxedBernoulli distribution.

        :param pi: Selection probability vector.
        :type pi: torch.Tensor
        :param R: Covariance matrix.
        :type R: torch.Tensor
        :param tau: Temperature hyper-parameter.
        :type tau: torch.Tensor
        :param validate_args: Whether to validate arguments.
        :type validate_args: bool
        """
        if validate_args:
            self._validate_args(pi, R, tau)

        self.pi = pi
        self.R = R
        self.tau = tau
        self.L = torch.linalg.cholesky(
            R
        )  # Cholesky decomposition of the covariance matrix
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `pi` is a tensor of shape (batch_size, pi_shape),
        the batch shape will be `[batch_size]`, indicating batch_size independent Bernoulli distributions.
        """
        return self.pi.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.
        For example, if `pi` is a tensor of shape (batch_size, pi_shape),
        the event shape will be `[pi_shape]`.
        """
        return self.pi.shape[-1:]

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution using the reparameterization trick.

        :param sample_shape: The shape of the sample.
        :type sample_shape: torch.Size
        :return: A sample from the distribution.
        :rtype: torch.Tensor
        """
        # Sample from the standard multivariate normal distribution
        shape = tuple(sample_shape) + tuple(self.pi.shape)
        eps = torch.randn(shape).to(self.pi.device)
        v = torch.einsum("...ij,...j->...i", self.L, eps)

        # Generate correlated uniform random variables
        uk = Normal(0, 1).cdf(v)

        # Generate relaxed multivariate Bernoulli variable
        log_pi = torch.log(self.pi)
        log_one_minus_pi = torch.log(1 - self.pi)
        log_uk = torch.log(uk)
        log_one_minus_uk = torch.log(1 - uk)

        m_tilde = torch.sigmoid(
            (log_pi - log_one_minus_pi + log_uk - log_one_minus_uk) / self.tau
        )

        return m_tilde

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution.

        :param sample_shape: The shape of the sample.
        :type sample_shape: torch.Size
        :return: A sample from the distribution.
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log probability of the given value.

        :param value: The value for which to compute the log probability.
        :type value: torch.Tensor
        :return: The log probability of the given value.
        :rtype: torch.Tensor
        """
        if self._validate_args:
            self._validate_sample(value)

        # Compute the log probability using the normal distribution
        log_prob = Normal(self.pi, self.tau).log_prob(value)

        # Adjust for the clipping to [0, 1]
        cdf_0 = Normal(self.pi, self.tau).cdf(torch.zeros_like(value))
        cdf_1 = Normal(self.pi, self.tau).cdf(torch.ones_like(value))
        log_prob = torch.where(value == 0, torch.log(cdf_0), log_prob)
        log_prob = torch.where(value == 1, torch.log(1 - cdf_1), log_prob)

        return log_prob

    def _validate_sample(self, value: torch.Tensor):
        r"""
        Validates the given sample value.

        :param value: The sample value to validate.
        :type value: torch.Tensor
        """
        if self._validate_args:
            if not (value >= 0).all() or not (value <= 1).all():
                raise ValueError("Sample value must be in the range [0, 1]")