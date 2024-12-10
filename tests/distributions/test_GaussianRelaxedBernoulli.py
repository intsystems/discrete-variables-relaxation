import torch
import sys, os

from relaxit.distributions import GaussianRelaxedBernoulli


def test_sample_shape():
    """
    This function tests the shape of samples generated from the GaussianRelaxedBernoulli distribution.

    Steps:
    1. Define the mean (loc) of the normal distribution.
    2. Define the standard deviation (scale) of the normal distribution.
    3. Create a GaussianRelaxedBernoulli distribution using the defined parameters.
    4. Generate samples from the distribution using the reparameterization trick with a specified sample shape.
    5. Assert that the shape of the generated samples matches the expected shape.
    """
    # Step 1: Define the mean (loc) of the normal distribution
    loc = torch.tensor([0.0])

    # Step 2: Define the standard deviation (scale) of the normal distribution
    scale = torch.tensor([1.0])

    # Step 3: Create a GaussianRelaxedBernoulli distribution using the defined parameters
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)

    # Step 4: Generate samples from the distribution using the reparameterization trick with a specified sample shape
    samples = distribution.rsample(sample_shape=torch.Size([3]))

    # Step 5: Assert that the shape of the generated samples matches the expected shape
    assert samples.shape == torch.Size([3, 1])


def test_sample_grad():
    """
    This function tests whether the samples generated from the GaussianRelaxedBernoulli distribution
    support gradient computation.

    Steps:
    1. Define the mean (loc) of the normal distribution with gradient tracking enabled.
    2. Define the standard deviation (scale) of the normal distribution with gradient tracking enabled.
    3. Create a GaussianRelaxedBernoulli distribution using the defined parameters.
    4. Generate samples from the distribution using the reparameterization trick.
    5. Assert that the generated samples support gradient computation.
    """
    # Step 1: Define the mean (loc) of the normal distribution with gradient tracking enabled
    loc = torch.tensor([0.0], requires_grad=True)

    # Step 2: Define the standard deviation (scale) of the normal distribution with gradient tracking enabled
    scale = torch.tensor([1.0], requires_grad=True)

    # Step 3: Create a GaussianRelaxedBernoulli distribution using the defined parameters
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)

    # Step 4: Generate samples from the distribution using the reparameterization trick
    samples = distribution.rsample()

    # Step 5: Assert that the generated samples support gradient computation
    assert samples.requires_grad == True


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the GaussianRelaxedBernoulli distribution.

    Steps:
    1. Define the mean (loc) of the normal distribution.
    2. Define the standard deviation (scale) of the normal distribution.
    3. Create a GaussianRelaxedBernoulli distribution using the defined parameters.
    4. Define a value for which to compute the log probability.
    5. Compute the log probability of the defined value.
    6. Assert that the shape of the computed log probability matches the expected shape.
    """
    # Step 1: Define the mean (loc) of the normal distribution
    loc = torch.tensor([0.0])

    # Step 2: Define the standard deviation (scale) of the normal distribution
    scale = torch.tensor([1.0])

    # Step 3: Create a GaussianRelaxedBernoulli distribution using the defined parameters
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)

    # Step 4: Define a value for which to compute the log probability
    value = torch.tensor([1.0])

    # Step 5: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 6: Assert that the shape of the computed log probability matches the expected shape
    assert log_prob.shape == torch.Size([1])


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the GaussianRelaxedBernoulli distribution
    supports gradient computation.

    Steps:
    1. Define the mean (loc) of the normal distribution with gradient tracking enabled.
    2. Define the standard deviation (scale) of the normal distribution with gradient tracking enabled.
    3. Create a GaussianRelaxedBernoulli distribution using the defined parameters.
    4. Define a value for which to compute the log probability.
    5. Compute the log probability of the defined value.
    6. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the mean (loc) of the normal distribution with gradient tracking enabled
    loc = torch.tensor([0.0], requires_grad=True)

    # Step 2: Define the standard deviation (scale) of the normal distribution with gradient tracking enabled
    scale = torch.tensor([1.0], requires_grad=True)

    # Step 3: Create a GaussianRelaxedBernoulli distribution using the defined parameters
    distribution = GaussianRelaxedBernoulli(loc=loc, scale=scale)

    # Step 4: Define a value for which to compute the log probability
    value = torch.tensor([1.0])

    # Step 5: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 6: Assert that the computed log probability supports gradient computation
    assert log_prob.requires_grad == True
