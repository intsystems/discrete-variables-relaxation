## Quickstart

```python
import torch
from relaxit.distributions.InvertibleGaussian import InvertibleGaussian

# initialize distribution parameters
loc = torch.zeros(3, 4, 5, requires_grad=True)
scale = torch.ones(3, 4, 5, requires_grad=True)
temperature = torch.tensor([1e-0])

# initialize distribution
distribution = InvertibleGaussian(loc, scale, temperature)

# sample with reparameterization
sample = distribution.rsample()
```