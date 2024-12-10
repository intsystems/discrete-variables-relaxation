import torch
import sys, os

from relaxit.distributions import StochasticTimesSmooth


def test_sample_shape():
    """
    This function tests the shape of samples generated from the StochasticTimesSmooth distribution.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StochasticTimesSmooth distribution using the defined logits.
    3. Generate samples from the distribution using the reparameterization trick.
    4. Assert that the shape of the generated samples matches the expected shape.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.tensor([1., 2., 3.])

    # Step 2: Create a StochasticTimesSmooth distribution using the defined logits
    distribution = StochasticTimesSmooth(logits=logits)

    # Step 3: Generate samples from the distribution using the reparameterization trick
    samples = distribution.rsample()

    # Step 4: Assert that the shape of the generated samples matches the expected shape
    assert samples.shape == torch.Size([3])


def test_sample_grad():
    """
    This function tests whether the samples generated from the StochasticTimesSmooth distribution
    support gradient computation.

    Steps:
    1. Define the logits for the distribution with gradient tracking enabled.
    2. Create a StochasticTimesSmooth distribution using the defined logits.
    3. Generate samples from the distribution using the reparameterization trick.
    4. Assert that the generated samples support gradient computation.
    """
    # Step 1: Define the logits for the distribution with gradient tracking enabled
    logits = torch.tensor([1., 2., 3.], requires_grad=True)

    # Step 2: Create a StochasticTimesSmooth distribution using the defined logits
    distribution = StochasticTimesSmooth(logits=logits)

    # Step 3: Generate samples from the distribution using the reparameterization trick
    samples = distribution.rsample()

    # Step 4: Assert that the generated samples support gradient computation
    assert samples.requires_grad == True


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the StochasticTimesSmooth distribution.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StochasticTimesSmooth distribution using the defined logits.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the shape of the computed log probability matches the expected shape.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.tensor([1., 2., 3.])

    # Step 2: Create a StochasticTimesSmooth distribution using the defined logits
    distribution = StochasticTimesSmooth(logits=logits)

    # Step 3: Define a value for which to compute the log probability
    value = torch.Tensor([1., 1., 1.])

    # Step 4: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 5: Assert that the shape of the computed log probability matches the expected shape
    assert log_prob.shape == torch.Size([3])


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the StochasticTimesSmooth distribution
    supports gradient computation.

    Steps:
    1. Define the logits for the distribution with gradient tracking enabled.
    2. Create a StochasticTimesSmooth distribution using the defined logits.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the logits for the distribution with gradient tracking enabled
    logits = torch.tensor([1., 2., 3.], requires_grad=True)

    # Step 2: Create a StochasticTimesSmooth distribution using the defined logits
    distribution = StochasticTimesSmooth(logits=logits)

    # Step 3: Define a value for which to compute the log probability
    value = torch.Tensor([1., 1., 1.])

    # Step 4: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 5: Assert that the computed log probability supports gradient computation
    assert log_prob.requires_grad == True
