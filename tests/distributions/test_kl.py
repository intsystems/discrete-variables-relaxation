import torch
import sys

from relaxit.distributions import InvertibleGaussian
from relaxit.distributions.kl import kl_divergence


def test_igr_kl_shape():
    """
    This function tests the shape of the KL-divergence computed between two InvertibleGaussian distributions.

    Steps:
    1. Define the mean (loc_1) and standard deviation (scale_1) of the first distribution with gradient tracking enabled.
    2. Define the temperature hyper-parameter (temperature_1) for the first distribution.
    3. Create the first InvertibleGaussian distribution using the defined parameters.
    4. Define the mean (loc_2) and standard deviation (scale_2) of the second distribution with gradient tracking enabled.
    5. Define the temperature hyper-parameter (temperature_2) for the second distribution.
    6. Create the second InvertibleGaussian distribution using the defined parameters.
    7. Compute the KL-divergence between the two distributions.
    8. Assert that the shape of the computed KL-divergence matches the expected shape.
    """
    # Step 1: Define the mean (loc_1) and standard deviation (scale_1) of the first distribution with gradient tracking enabled
    loc_1 = torch.zeros(3, 4, 5, requires_grad=True)
    scale_1 = torch.ones(3, 4, 5, requires_grad=True)

    # Step 2: Define the temperature hyper-parameter (temperature_1) for the first distribution
    temperature_1 = torch.tensor([1e-0])

    # Step 3: Create the first InvertibleGaussian distribution using the defined parameters
    dist_1 = InvertibleGaussian(loc_1, scale_1, temperature_1)

    # Step 4: Define the mean (loc_2) and standard deviation (scale_2) of the second distribution with gradient tracking enabled
    loc_2 = torch.ones(3, 4, 5, requires_grad=True)  # ones, not zeros
    scale_2 = torch.ones(3, 4, 5, requires_grad=True)

    # Step 5: Define the temperature hyper-parameter (temperature_2) for the second distribution
    temperature_2 = torch.tensor([1e-2])

    # Step 6: Create the second InvertibleGaussian distribution using the defined parameters
    dist_2 = InvertibleGaussian(loc_2, scale_2, temperature_2)

    # Step 7: Compute the KL-divergence between the two distributions
    div = kl_divergence(dist_1, dist_2)

    # Step 8: Assert that the shape of the computed KL-divergence matches the expected shape
    assert div.shape == torch.Size([3, 4, 5])


def test_igr_kl_grad():
    """
    This function tests whether the KL-divergence computed between two InvertibleGaussian distributions
    supports gradient computation.

    Steps:
    1. Define the mean (loc_1) and standard deviation (scale_1) of the first distribution with gradient tracking enabled.
    2. Define the temperature hyper-parameter (temperature_1) for the first distribution.
    3. Create the first InvertibleGaussian distribution using the defined parameters.
    4. Define the mean (loc_2) and standard deviation (scale_2) of the second distribution with gradient tracking enabled.
    5. Define the temperature hyper-parameter (temperature_2) for the second distribution.
    6. Create the second InvertibleGaussian distribution using the defined parameters.
    7. Compute the KL-divergence between the two distributions.
    8. Assert that the computed KL-divergence supports gradient computation.
    """
    # Step 1: Define the mean (loc_1) and standard deviation (scale_1) of the first distribution with gradient tracking enabled
    loc_1 = torch.zeros(3, 4, 5, requires_grad=True)
    scale_1 = torch.ones(3, 4, 5, requires_grad=True)

    # Step 2: Define the temperature hyper-parameter (temperature_1) for the first distribution
    temperature_1 = torch.tensor([1e-0])

    # Step 3: Create the first InvertibleGaussian distribution using the defined parameters
    dist_1 = InvertibleGaussian(loc_1, scale_1, temperature_1)

    # Step 4: Define the mean (loc_2) and standard deviation (scale_2) of the second distribution with gradient tracking enabled
    loc_2 = torch.ones(3, 4, 5, requires_grad=True)  # ones, not zeros
    scale_2 = torch.ones(3, 4, 5, requires_grad=True)

    # Step 5: Define the temperature hyper-parameter (temperature_2) for the second distribution
    temperature_2 = torch.tensor([1e-2])

    # Step 6: Create the second InvertibleGaussian distribution using the defined parameters
    dist_2 = InvertibleGaussian(loc_2, scale_2, temperature_2)

    # Step 7: Compute the KL-divergence between the two distributions
    div = kl_divergence(dist_1, dist_2)

    # Step 8: Assert that the computed KL-divergence supports gradient computation
    assert div.requires_grad == True


def test_igr_kl_value():
    """
    This function tests the value of the KL-divergence computed between two identical InvertibleGaussian distributions.

    Steps:
    1. Define the mean (loc_1) and standard deviation (scale_1) of the first distribution with gradient tracking enabled.
    2. Define the temperature hyper-parameter (temperature_1) for the first distribution.
    3. Create the first InvertibleGaussian distribution using the defined parameters.
    4. Define the mean (loc_2) and standard deviation (scale_2) of the second distribution with gradient tracking enabled.
    5. Define the temperature hyper-parameter (temperature_2) for the second distribution.
    6. Create the second InvertibleGaussian distribution using the defined parameters.
    7. Compute the KL-divergence between the two distributions.
    8. Assert that the computed KL-divergence is close to zero.
    """
    # Step 1: Define the mean (loc_1) and standard deviation (scale_1) of the first distribution with gradient tracking enabled
    loc_1 = torch.ones(3, 4, 5, requires_grad=True)
    scale_1 = torch.ones(3, 4, 5, requires_grad=True)

    # Step 2: Define the temperature hyper-parameter (temperature_1) for the first distribution
    temperature_1 = torch.tensor([1e-2])

    # Step 3: Create the first InvertibleGaussian distribution using the defined parameters
    dist_1 = InvertibleGaussian(loc_1, scale_1, temperature_1)

    # Step 4: Define the mean (loc_2) and standard deviation (scale_2) of the second distribution with gradient tracking enabled
    loc_2 = torch.ones(3, 4, 5, requires_grad=True)  # ones, not zeros
    scale_2 = torch.ones(3, 4, 5, requires_grad=True)

    # Step 5: Define the temperature hyper-parameter (temperature_2) for the second distribution
    temperature_2 = torch.tensor([1e-2])

    # Step 6: Create the second InvertibleGaussian distribution using the defined parameters
    dist_2 = InvertibleGaussian(loc_2, scale_2, temperature_2)

    # Step 7: Compute the KL-divergence between the two distributions
    div = kl_divergence(dist_1, dist_2)

    # Step 8: Assert that the computed KL-divergence is close to zero
    assert torch.allclose(div, torch.zeros_like(div))
