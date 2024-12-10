import torch
from pyro.distributions import constraints, Normal
from pyro.distributions.torch import TransformedDistribution
from pyro.distributions.transforms import SoftmaxTransform


class LogisticNormalSoftmax(TransformedDistribution):
    r"""
    Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Normal` distribution transformed with the
    `SoftmaxTransform` such that:

        X ~ LogisticNormal(loc, scale)
        Y = Logistic(X) ~ Normal(loc, scale)

    Args:
        loc (float or torch.Tensor): Mean of the base distribution.
        scale (float or torch.Tensor): Standard deviation of the base distribution.
        validate_args (bool, optional): Whether to validate arguments. Defaults to None.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        r"""
        Initializes the LogisticNormalSoftmax distribution.

        Args:
            loc (float or torch.Tensor): Mean of the base distribution.
            scale (float or torch.Tensor): Standard deviation of the base distribution.
            validate_args (bool, optional): Whether to validate arguments. Defaults to None.
        """
        base_dist = Normal(loc, scale, validate_args=validate_args)
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        super().__init__(base_dist, SoftmaxTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        r"""
        Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch
        dimensions expanded to `batch_shape`.

        Args:
            batch_shape (torch.Size): The desired expanded size.
            _instance (LogisticNormalSoftmax, optional): New instance of the distribution to populate. Defaults to None.

        Returns:
            LogisticNormalSoftmax: New distribution instance with batch dimensions expanded to `batch_shape`.
        """
        new = self._get_checked_instance(LogisticNormalSoftmax, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        r"""
        Returns the location (mean) of the base distribution.

        Returns:
            float or torch.Tensor: The location of the base distribution.
        """
        return self.base_dist.base_dist.loc

    @property
    def scale(self):
        r"""
        Returns the scale (standard deviation) of the base distribution.

        Returns:
            float or torch.Tensor: The scale of the base distribution.
        """
        return self.base_dist.base_dist.scale
