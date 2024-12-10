import torch
import sys

from relaxit.distributions import InvertibleGaussian


def test_sample_shape():
    """
    This function tests the shape of samples generated from the InvertibleGaussian distribution.

    Steps:
    1. Define the mean (loc) of the normal distribution.
    2. Define the standard deviation (scale) of the normal distribution.
    3. Define the temperature hyper-parameter.
    4. Create an InvertibleGaussian distribution using the defined parameters.
    5. Generate a sample from the distribution using the reparameterization trick.
    6. Assert that the shape of the generated sample matches the expected shape.
    """
    # Step 1: Define the mean (loc) of the normal distribution
    loc = torch.zeros(3, 4, 5)

    # Step 2: Define the standard deviation (scale) of the normal distribution
    scale = torch.ones(3, 4, 5)

    # Step 3: Define the temperature hyper-parameter
    temperature = torch.tensor([1e-0])

    # Step 4: Create an InvertibleGaussian distribution using the defined parameters
    distribution = InvertibleGaussian(loc, scale, temperature)

    # Step 5: Generate a sample from the distribution using the reparameterization trick
    sample = distribution.rsample()

    # Step 6: Assert that the shape of the generated sample matches the expected shape
    assert sample.shape == torch.Size([3, 4, 6])


def test_sample_grad():
    """
    This function tests whether the samples generated from the InvertibleGaussian distribution
    support gradient computation.

    Steps:
    1. Define the mean (loc) of the normal distribution with gradient tracking enabled.
    2. Define the standard deviation (scale) of the normal distribution with gradient tracking enabled.
    3. Define the temperature hyper-parameter.
    4. Create an InvertibleGaussian distribution using the defined parameters.
    5. Generate a sample from the distribution using the reparameterization trick.
    6. Assert that the generated sample supports gradient computation.
    """
    # Step 1: Define the mean (loc) of the normal distribution with gradient tracking enabled
    loc = torch.zeros(3, 4, 5, requires_grad=True)

    # Step 2: Define the standard deviation (scale) of the normal distribution with gradient tracking enabled
    scale = torch.ones(3, 4, 5, requires_grad=True)

    # Step 3: Define the temperature hyper-parameter
    temperature = torch.tensor([1e-0])

    # Step 4: Create an InvertibleGaussian distribution using the defined parameters
    distribution = InvertibleGaussian(loc, scale, temperature)

    # Step 5: Generate a sample from the distribution using the reparameterization trick
    sample = distribution.rsample()

    # Step 6: Assert that the generated sample supports gradient computation
    assert sample.requires_grad == True


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the InvertibleGaussian distribution.

    Steps:
    1. Define the mean (loc) of the normal distribution.
    2. Define the standard deviation (scale) of the normal distribution.
    3. Define the temperature hyper-parameter.
    4. Create an InvertibleGaussian distribution using the defined parameters.
    5. Define a value for which to compute the log probability.
    6. Compute the log probability of the defined value.
    7. Assert that the shape of the computed log probability matches the expected shape.
    """
    # Step 1: Define the mean (loc) of the normal distribution
    loc = torch.zeros(3, 4, 5)

    # Step 2: Define the standard deviation (scale) of the normal distribution
    scale = torch.ones(3, 4, 5)

    # Step 3: Define the temperature hyper-parameter
    temperature = torch.tensor([1e-0])

    # Step 4: Create an InvertibleGaussian distribution using the defined parameters
    distribution = InvertibleGaussian(loc, scale, temperature)

    # Step 5: Define a value for which to compute the log probability
    value = 0.5 * torch.ones(3, 4, 6)

    # Step 6: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 7: Assert that the shape of the computed log probability matches the expected shape
    assert log_prob.shape == torch.Size([3, 4, 5])


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the InvertibleGaussian distribution
    supports gradient computation.

    Steps:
    1. Define the mean (loc) of the normal distribution with gradient tracking enabled.
    2. Define the standard deviation (scale) of the normal distribution with gradient tracking enabled.
    3. Define the temperature hyper-parameter.
    4. Create an InvertibleGaussian distribution using the defined parameters.
    5. Define a value for which to compute the log probability.
    6. Compute the log probability of the defined value.
    7. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the mean (loc) of the normal distribution with gradient tracking enabled
    loc = torch.zeros(3, 4, 5, requires_grad=True)

    # Step 2: Define the standard deviation (scale) of the normal distribution with gradient tracking enabled
    scale = torch.ones(3, 4, 5, requires_grad=True)

    # Step 3: Define the temperature hyper-parameter
    temperature = torch.tensor([1e-0])

    # Step 4: Create an InvertibleGaussian distribution using the defined parameters
    distribution = InvertibleGaussian(loc, scale, temperature)

    # Step 5: Define a value for which to compute the log probability
    value = 0.5 * torch.ones(3, 4, 6)

    # Step 6: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 7: Assert that the computed log probability supports gradient computation
    assert log_prob.requires_grad == True
