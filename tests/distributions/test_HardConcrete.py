import torch
import sys, os

from relaxit.distributions import HardConcrete


def test_sample_shape():
    """
    This function tests the shape of samples generated from the HardConcrete distribution.

    Steps:
    1. Define the parameters alpha, beta, gamma, and xi for the distribution.
    2. Create a HardConcrete distribution using the defined parameters.
    3. Generate samples from the distribution using the reparameterization trick with a specified sample shape.
    4. Assert that the shape of the generated samples matches the expected shape.
    """
    # Step 1: Define the parameters alpha, beta, gamma, and xi for the distribution
    alpha = torch.tensor([1.0])
    beta = torch.tensor([2.0])
    gamma = torch.tensor([-3.0])
    xi = torch.tensor([4.0])

    # Step 2: Create a HardConcrete distribution using the defined parameters
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)

    # Step 3: Generate samples from the distribution using the reparameterization trick with a specified sample shape
    samples = distribution.rsample(sample_shape=torch.Size([3]))

    # Step 4: Assert that the shape of the generated samples matches the expected shape
    assert samples.shape == torch.Size([3, 1])


def test_sample_grad():
    """
    This function tests whether the samples generated from the HardConcrete distribution
    support gradient computation.

    Steps:
    1. Define the parameters alpha, beta, gamma, and xi for the distribution with gradient tracking enabled.
    2. Create a HardConcrete distribution using the defined parameters.
    3. Generate samples from the distribution using the reparameterization trick with a specified sample shape.
    4. Assert that the generated samples support gradient computation.
    """
    # Step 1: Define the parameters alpha, beta, gamma, and xi for the distribution with gradient tracking enabled
    alpha = torch.tensor([1.0], requires_grad=True)
    beta = torch.tensor([2.0], requires_grad=True)
    gamma = torch.tensor([-3.0], requires_grad=True)
    xi = torch.tensor([4.0], requires_grad=True)

    # Step 2: Create a HardConcrete distribution using the defined parameters
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)

    # Step 3: Generate samples from the distribution using the reparameterization trick with a specified sample shape
    samples = distribution.rsample(sample_shape=torch.Size([3]))

    # Step 4: Assert that the generated samples support gradient computation
    assert samples.requires_grad == True


def test_log_prob_shape():
    """
    This function tests the shape of the log probability computed from the HardConcrete distribution.

    Steps:
    1. Define the parameters alpha, beta, gamma, and xi for the distribution.
    2. Create a HardConcrete distribution using the defined parameters.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the shape of the computed log probability matches the expected shape.
    """
    # Step 1: Define the parameters alpha, beta, gamma, and xi for the distribution
    alpha = torch.tensor([1.0])
    beta = torch.tensor([2.0])
    gamma = torch.tensor([-3.0])
    xi = torch.tensor([4.0])

    # Step 2: Create a HardConcrete distribution using the defined parameters
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)

    # Step 3: Define a value for which to compute the log probability
    value = torch.tensor([1.0])

    # Step 4: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 5: Assert that the shape of the computed log probability matches the expected shape
    assert log_prob.shape == torch.Size([1])


def test_log_prob_grad():
    """
    This function tests whether the log probability computed from the HardConcrete distribution
    supports gradient computation.

    Steps:
    1. Define the parameters alpha, beta, gamma, and xi for the distribution with gradient tracking enabled.
    2. Create a HardConcrete distribution using the defined parameters.
    3. Define a value for which to compute the log probability.
    4. Compute the log probability of the defined value.
    5. Assert that the computed log probability supports gradient computation.
    """
    # Step 1: Define the parameters alpha, beta, gamma, and xi for the distribution with gradient tracking enabled
    alpha = torch.tensor([1.0], requires_grad=True)
    beta = torch.tensor([2.0], requires_grad=True)
    gamma = torch.tensor([-3.0], requires_grad=True)
    xi = torch.tensor([4.0], requires_grad=True)

    # Step 2: Create a HardConcrete distribution using the defined parameters
    distribution = HardConcrete(alpha=alpha, beta=beta, gamma=gamma, xi=xi)

    # Step 3: Define a value for which to compute the log probability
    value = torch.tensor([1.0])

    # Step 4: Compute the log probability of the defined value
    log_prob = distribution.log_prob(value)

    # Step 5: Assert that the computed log probability supports gradient computation
    assert log_prob.requires_grad == True
