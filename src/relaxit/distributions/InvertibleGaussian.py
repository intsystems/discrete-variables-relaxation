import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal


class InvertibleGaussian(TorchDistribution):
    r"""
    Invertible Gaussian distribution class from https://arxiv.org/abs/1912.09588.

    Args:
        loc (torch.Tensor): The mean (mu) of the normal distribution.
        scale (torch.Tensor): The standard deviation (sigma) of the normal distribution.
        temperature (float): Temperature parameter for the softmax++ function.
        validate_args (bool, optional): Whether to validate arguments. Defaults to None.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, temperature, validate_args: bool = None):
        r"""
        Initializes the Invertible Gaussian distribution.

        Args:
            loc (torch.Tensor): Mean of the normal distribution.
            scale (torch.Tensor): Standard deviation of the normal distribution.
            temperature (float): Temperature parameter for the softmax++ function.
            validate_args (bool, optional): Whether to validate arguments. Defaults to None.

        The batch shape is inferred from the shape of the parameters (loc and scale),
        meaning it defines how many independent distributions are parameterized.
        """
        self.loc = loc
        self.scale = scale
        self.temperature = temperature
        batch_shape = torch.Size() if loc.dim() == 0 else loc.shape
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        r"""
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.

        Returns:
            torch.Size: The batch shape of the distribution.
        """
        return self.loc.shape

    @property
    def event_shape(self) -> torch.Size:
        r"""
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.

        Returns:
            torch.Size: The event shape of the distribution.
        """
        return torch.Size()

    def softmax_plus_plus(self, y: torch.Tensor, delta: float = 1) -> torch.Tensor:
        r"""
        Computes the softmax++ function.

        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, num_classes).
            delta (float, optional): Additional term delta > 0. Defaults to 1.

        Returns:
            torch.Tensor: Output tensor of the same shape as y.
        """
        # Scale the input by the temperature
        scaled_y = y / self.temperature

        # Compute the exponentials
        exp_y = torch.exp(scaled_y)

        # Compute the denominator
        denominator = torch.sum(exp_y, dim=-1, keepdim=True) + delta

        # Compute the softmax++
        softmax_pp = exp_y / denominator

        return softmax_pp

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        r"""
        Generates a sample from the distribution using the reparameterization trick.

        Args:
            sample_shape (torch.Size, optional): The shape of the generated samples. Defaults to torch.Size().

        Returns:
            torch.Tensor: A sample from the distribution.
        """
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        y = self.loc + self.scale * eps
        g = self.softmax_plus_plus(y)
        residual = 1 - torch.sum(g, dim=-1, keepdim=True)
        return torch.cat([g, residual], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the log likelihood of a value.

        Args:
            value (torch.Tensor): The value for which to compute the log probability.

        Returns:
            torch.Tensor: The log probability of the given value.
        """
        if self._validate_args:
            self._validate_sample(value)

        # Separate the value into g and residual
        g = value[..., :-1]
        residual = value[..., -1:]

        # Invert the softmax++ transformation
        y = self.temperature * (torch.log(g) - torch.log(1 - residual))

        # Compute the log probability of the normal distribution
        log_prob_normal = -0.5 * (
            ((y - self.loc) / self.scale) ** 2
            + torch.log(2 * torch.tensor(torch.pi))
            + 2 * torch.log(self.scale)
        )

        # Compute the Jacobian determinant of the softmax++ transformation
        K = g.size(-1) + 1  # Number of classes including the residual
        log_det_jacobian = (
            -(K - 1) * torch.log(self.temperature).item()
            + torch.sum(torch.log(g), dim=-1, keepdim=True)
            + torch.log(residual)
        )

        # Adjust the log probability by the Jacobian determinant
        log_prob = log_prob_normal - log_det_jacobian

        return log_prob

    def _validate_sample(self, value: torch.Tensor):
        r"""
        Validates the given sample value.

        Args:
            value (torch.Tensor): The sample value to validate.
        """
        if self._validate_args:
            if not (value >= 0).all() or not (value <= 1).all():
                raise ValueError("Sample value must be in the range [0, 1]")
