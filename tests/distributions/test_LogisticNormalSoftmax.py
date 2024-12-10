import torch
import sys

from relaxit.distributions import LogisticNormalSoftmax


def test_sample_shape():
    """
    This function tests the shape of samples generated from the LogisticNormalSoftmax distribution.

    Steps:
    1. Define the mean (loc) of the normal distribution.
    2. Define the standard deviation (scale) of the normal distribution.
    3. Create a LogisticNormalSoftmax distribution using the defined parameters.
    4. Generate a sample from the distribution using the reparameterization trick.
    5. Assert that the shape of the generated sample matches the expected shape.
    """
    # Step 1: Define the mean (loc) of the normal distribution
    loc = torch.zeros(3, 4, 5)

    # Step 2: Define the standard deviation (scale) of the normal distribution
    scale = torch.ones(3, 4, 5)

    # Step 3: Create a LogisticNormalSoftmax distribution using the defined parameters
    distribution = LogisticNormalSoftmax(loc, scale)

    # Step 4: Generate a sample from the distribution using the reparameterization trick
    sample = distribution.rsample()

    # Step 5: Assert that the shape of the generated sample matches the expected shape
    assert sample.shape == torch.Size([3, 4, 5])


def test_sample_grad():
    """
    This function tests whether the samples generated from the LogisticNormalSoftmax distribution
    support gradient computation.

    Steps:
    1. Define the mean (loc) of the normal distribution with gradient tracking enabled.
    2. Define the standard deviation (scale) of the normal distribution with gradient tracking enabled.
    3. Create a LogisticNormalSoftmax distribution using the defined parameters.
    4. Generate a sample from the distribution using the reparameterization trick.
    5. Assert that the generated sample supports gradient computation.
    """
    # Step 1: Define the mean (loc) of the normal distribution with gradient tracking enabled
    loc = torch.zeros(3, 4, 5, requires_grad=True)

    # Step 2: Define the standard deviation (scale) of the normal distribution with gradient tracking enabled
    scale = torch.ones(3, 4, 5, requires_grad=True)

    # Step 3: Create a LogisticNormalSoftmax distribution using the defined parameters
    distribution = LogisticNormalSoftmax(loc, scale)

    # Step 4: Generate a sample from the distribution using the reparameterization trick
    sample = distribution.rsample()

    # Step 5: Assert that the generated sample supports gradient computation
    assert sample.requires_grad == True
