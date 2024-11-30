## Quickstart
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intsystems/discrete-variables-relaxation/blob/main/demo/quickstart.ipynb)
```python
import torch
from relaxit.distributions import InvertibleGaussian

# initialize distribution parameters
loc = torch.zeros(3, 4, 5, requires_grad=True)
scale = torch.ones(3, 4, 5, requires_grad=True)
temperature = torch.tensor([1e-0])

# initialize distribution
distribution = InvertibleGaussian(loc, scale, temperature)

# sample with reparameterization
sample = distribution.rsample()
print('sample.shape:', sample.shape)
print('sample.requires_grad:', sample.requires_grad)
```