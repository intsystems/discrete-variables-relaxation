from pyro.distributions import constraints, Normal
from pyro.distributions.torch import TransformedDistribution
from pyro.distributions.transforms import SoftmaxTransform


class LogisticNormalSoftmax(TransformedDistribution):
    r"""
    Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Normal` distribution transformed with the
    `SoftmaxTransform` such that::

        X ~ LogisticNormal(loc, scale)
        Y = Logistic(X) ~ Normal(loc, scale)

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        super().__init__(base_dist, SoftmaxTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticNormalSoftmax, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.base_dist.scale
