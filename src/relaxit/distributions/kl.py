from torch.distributions import kl_divergence, register_kl, Normal

from .InvertibleGaussian import InvertibleGaussian


@register_kl(InvertibleGaussian, InvertibleGaussian)
def _kl_igr_igr(p: InvertibleGaussian, q: InvertibleGaussian):
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.
    
    Based on the paper https://arxiv.org/abs/1912.09588.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (InvertibleGaussian): A :class:`~relaxit.distributions.InvertibleGaussian` object.
        q (InvertibleGaussian): A :class:`~relaxit.distributions.InvertibleGaussian` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.
    """
    p_normal = Normal(p.loc, p.scale)
    q_normal = Normal(q.loc, q.scale)
    return kl_divergence(p_normal, q_normal)
