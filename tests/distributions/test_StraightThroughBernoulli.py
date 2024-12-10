import torch
import sys, os

from relaxit.distributions import StraightThroughBernoulli


def test_sample_shape():
    """
    This function tests the shape of samples generated from the StraightThroughBernoulli distribution.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StraightThroughBernoulli distribution using the defined logits.
    3. Generate a sample from the distribution using the reparameterization trick.
    4. Assert that the shape of the generated sample matches the shape of the logits.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.tensor([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])

    # Step 2: Create a StraightThroughBernoulli distribution using the defined logits
    distribution = StraightThroughBernoulli(logits=logits)

    # Step 3: Generate a sample from the distribution using the reparameterization trick
    sample = distribution.rsample()

    # Step 4: Assert that the shape of the generated sample matches the shape of the logits
    assert sample.shape == logits.shape


def test_sample_grad():
    """
    This function tests whether the samples generated from the StraightThroughBernoulli distribution
    support gradient computation.

    Steps:
    1. Define the logits for the distribution with gradient tracking enabled.
    2. Create a StraightThroughBernoulli distribution using the defined logits.
    3. Generate a sample from the distribution using the reparameterization trick.
    4. Assert that the generated sample supports gradient computation.
    """
    # Step 1: Define the logits for the distribution with gradient tracking enabled
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

    # Step 2: Create a StraightThroughBernoulli distribution using the defined logits
    distribution = StraightThroughBernoulli(logits=logits)

    # Step 3: Generate a sample from the distribution using the reparameterization trick
    sample = distribution.rsample()

    # Step 4: Assert that the generated sample supports gradient computation
    assert sample.requires_grad == True


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the StraightThroughBernoulli distribution.

    Steps:
    1. Define the logits for the distribution.
    2. Create a StraightThroughBernoulli distribution using the defined logits.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the shape of the computed log probability matches the expected shape.
    """
    # Step 1: Define the logits for the distribution
    logits = torch.tensor([1.0, 2.0, 3.0])

    # Step 2: Create a StraightThroughBernoulli distribution using the defined logits
    distribution = StraightThroughBernoulli(logits=logits)

    # Step 3: Define a value for which to compute the log probability
    value = torch.tensor([1.0, 1.0, 1.0])

    # Step 4: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 5: Assert that the shape of the computed log probability matches the expected shape
    assert log_prob.shape == torch.Size([3])


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the StraightThroughBernoulli distribution
    supports gradient computation.

    Steps:
    1. Define the logits for the distribution with gradient tracking enabled.
    2. Create a StraightThroughBernoulli distribution using the defined logits.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the logits for the distribution with gradient tracking enabled
    logits = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Step 2: Create a StraightThroughBernoulli distribution using the defined logits
    distribution = StraightThroughBernoulli(logits=logits)

    # Step 3: Define a value for which to compute the log probability
    value = torch.tensor([1.0, 1.0, 1.0])

    # Step 4: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 5: Assert that the computed log probability supports gradient computation
    assert log_prob.requires_grad == True
