import torch
import sys, os

from relaxit.distributions import CorrelatedRelaxedBernoulli


def test_sample_shape():
    """
    This function tests the shape of samples generated from the CorrelatedRelaxedBernoulli distribution.

    Steps:
    1. Define the selection probability vector `pi`.
    2. Define the covariance matrix `R`.
    3. Define the temperature hyper-parameter `tau`.
    4. Create a CorrelatedRelaxedBernoulli distribution using the defined parameters.
    5. Generate samples from the distribution using the reparameterization trick.
    6. Assert that the shape of the generated samples matches the expected shape.
    """
    # Step 1: Define the selection probability vector `pi`
    pi = torch.tensor([0.1, 0.2, 0.3])

    # Step 2: Define the covariance matrix `R`
    R = torch.tensor([[1.0]])

    # Step 3: Define the temperature hyper-parameter `tau`
    tau = torch.tensor([2.0])

    # Step 4: Create a CorrelatedRelaxedBernoulli distribution using the defined parameters
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)

    # Step 5: Generate samples from the distribution using the reparameterization trick
    samples = distribution.rsample()

    # Step 6: Assert that the shape of the generated samples matches the expected shape
    assert samples.shape == torch.Size([3])


def test_sample_grad():
    """
    This function tests whether the samples generated from the CorrelatedRelaxedBernoulli distribution
    support gradient computation.

    Steps:
    1. Define the selection probability vector `pi` with gradient tracking enabled.
    2. Define the covariance matrix `R`.
    3. Define the temperature hyper-parameter `tau`.
    4. Create a CorrelatedRelaxedBernoulli distribution using the defined parameters.
    5. Generate samples from the distribution using the reparameterization trick.
    6. Assert that the generated samples support gradient computation.
    """
    # Step 1: Define the selection probability vector `pi` with gradient tracking enabled
    pi = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

    # Step 2: Define the covariance matrix `R`
    R = torch.tensor([[1.0]])

    # Step 3: Define the temperature hyper-parameter `tau`
    tau = torch.tensor([2.0])

    # Step 4: Create a CorrelatedRelaxedBernoulli distribution using the defined parameters
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)

    # Step 5: Generate samples from the distribution using the reparameterization trick
    samples = distribution.rsample()

    # Step 6: Assert that the generated samples support gradient computation
    assert samples.requires_grad == True


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the CorrelatedRelaxedBernoulli distribution.

    Steps:
    1. Define the selection probability vector `pi`.
    2. Define the covariance matrix `R`.
    3. Define the temperature hyper-parameter `tau`.
    4. Create a CorrelatedRelaxedBernoulli distribution using the defined parameters.
    5. Define a value for which to compute the log probability.
    6. Compute the log probability of the defined value.
    7. Assert that the shape of the computed log probability matches the expected shape.
    """
    # Step 1: Define the selection probability vector `pi`
    pi = torch.tensor([0.1, 0.2, 0.3])

    # Step 2: Define the covariance matrix `R`
    R = torch.tensor([[1.0]])

    # Step 3: Define the temperature hyper-parameter `tau`
    tau = torch.tensor([2.0])

    # Step 4: Create a CorrelatedRelaxedBernoulli distribution using the defined parameters
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)

    # Step 5: Define a value for which to compute the log probability
    value = torch.tensor([1.0])

    # Step 6: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 7: Assert that the shape of the computed log probability matches the expected shape
    assert log_prob.shape == torch.Size([3])


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the CorrelatedRelaxedBernoulli distribution
    supports gradient computation.

    Steps:
    1. Define the selection probability vector `pi` with gradient tracking enabled.
    2. Define the covariance matrix `R`.
    3. Define the temperature hyper-parameter `tau`.
    4. Create a CorrelatedRelaxedBernoulli distribution using the defined parameters.
    5. Define a value for which to compute the log probability.
    6. Compute the log probability of the defined value.
    7. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the selection probability vector `pi` with gradient tracking enabled
    pi = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

    # Step 2: Define the covariance matrix `R`
    R = torch.tensor([[1.0]])

    # Step 3: Define the temperature hyper-parameter `tau`
    tau = torch.tensor([2.0])

    # Step 4: Create a CorrelatedRelaxedBernoulli distribution using the defined parameters
    distribution = CorrelatedRelaxedBernoulli(pi=pi, R=R, tau=tau)

    # Step 5: Define a value for which to compute the log probability
    value = torch.tensor([1.0])

    # Step 6: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 7: Assert that the computed log probability supports gradient computation
    assert log_prob.requires_grad == True
