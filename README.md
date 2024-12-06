<div align="center">  
    <img src="assets/logo.svg" width="200px" />
    <h1> Just Relax It </h1>
    <p align="center"> Discrete Variables Relaxation </p>
</div>

<p align="center">
    <a href="https://pytorch.org/docs/stable/distributions.html">
        <img alt="Compatible with PyTorch" src="https://img.shields.io/badge/Compatible_with_Pytorch-ef4c2c">
    </a>
    <a href="https://docs.pyro.ai/en/dev/distributions.html">
        <img alt="Inspired by Pyro" src="https://img.shields.io/badge/Inspired_by_Pyro-fecd08">
    </a>
</p>

<p align="center">
    <a href="https://github.com/intsystems/discrete-variables-relaxation/tree/main/tests">
        <img alt="Coverage_2" src="https://github.com/intsystems/discrete-variables-relaxation/actions/workflows/test.yml/badge.svg" />
    </a>
    <a href="https://github.com/intsystems/discrete-variables-relaxation/tree/main/tests">
        <img alt="Coverage" src="coverage-badge.svg" />
    </a>
    <a href="https://intsystems.github.io/discrete-variables-relaxation">
        <img alt="Docs" src="https://github.com/intsystems/discrete-variables-relaxation/actions/workflows/docs.yml/badge.svg" />
    </a>
</p>

<p align="center">
    <a href="https://github.com/intsystems/discrete-variables-relaxation/blob/main/LICENSE">
        <img alt="License" src="https://badgen.net/github/license/intsystems/discrete-variables-relaxation?color=green" />
    </a>
    <a href="https://github.com/intsystems/discrete-variables-relaxation/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/intsystems/discrete-variables-relaxation" />
    </a>
    <a href="https://github.com/intsystems/discrete-variables-relaxation/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues-closed/intsystems/discrete-variables-relaxation?color=0088ff" />
    </a>
    <a href="https://github.com/intsystems/discrete-variables-relaxation/pulls">
        <img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr-closed/intsystems/discrete-variables-relaxation?color=7f29d6" />
    </a>
</p>

"Just Relax It" is a cutting-edge Python library designed to streamline the optimization of discrete probability distributions in neural networks, offering a suite of advanced relaxation techniques compatible with PyTorch.

## 📬 Assets

1. [Technical Meeting 1 - Presentation](https://github.com/intsystems/discrete-variables-relaxation/blob/main/assets/presentation_tm1.pdf)
2. [Technical Meeting 2 - Jupyter Notebook](https://github.com/intsystems/discrete-variables-relaxation/blob/main/basic/basic_code.ipynb)
3. [Technical Meeting 3 - Jupyter Notebook](https://github.com/intsystems/discrete-variables-relaxation/blob/main/demo/vizualization.ipynb)
4. [Blog Post](https://github.com/intsystems/discrete-variables-relaxation/blob/main/assets/blog-post.pdf)
5. [Documentation](https://intsystems.github.io/discrete-variables-relaxation/)
6. [Tests](https://github.com/intsystems/discrete-variables-relaxation/tree/main/tests)

## 💡 Motivation
For lots of mathematical problems we need an ability to sample discrete random variables.
The problem is that due to continuous nature of deep learning optimization, the usage of truly discrete random variables is infeasible.
Thus we use different relaxation methods. 
One of them, [Concrete distribution](https://arxiv.org/abs/1611.00712) or [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) (this is one distribution proposed in parallel by two research groups) is implemented in different DL packages. 
In this project we implement different alternatives to it. 
<div align="center">  
    <img src="assets/overview.png" width="600"/>
</div>

## 🗃 Algorithms
- [x] [Relaxed Bernoulli](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/GaussianRelaxedBernoulli.py), also see [📝 paper](http://proceedings.mlr.press/v119/yamada20a/yamada20a.pdf) 
- [x] [Correlated relaxed Bernoulli](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/CorrelatedRelaxedBernoulli.py), also see [📝 paper](https://openreview.net/pdf?id=oDFvtxzPOx)
- [x] [Gumbel-softmax TOP-K](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/GumbelSoftmaxTopK.py), also see [📝 paper](https://arxiv.org/pdf/1903.06059) 
- [x] [Straight-Through Bernoulli](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/StraightThroughBernoulli.py), also see [📝 paper](https://arxiv.org/abs/1910.02176) 
- [x] [Stochastic Times Smooth](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/StochasticTimesSmooth.py), also see [📝 paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=62c76ca0b2790c34e85ba1cce09d47be317c7235) 
- [x] [Invertible Gaussian](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/InvertibleGaussian.py) with [KL implemented](https://github.com/intsystems/discrete-variables-relaxation/blob/f398ebbbac703582de392bc33d89b55c6c99ea68/src/relaxit/distributions/kl.py#L7), also see [📝 paper](https://arxiv.org/abs/1912.09588)
- [x] [Hard Concrete](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/HardConcrete.py), also see [📝 paper](https://arxiv.org/pdf/1712.01312) 
- [x] [Logistic-Normal](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/LogisticNormalSoftmax.py) and [Laplace-form approximation of Dirichlet](https://github.com/intsystems/discrete-variables-relaxation/blob/main/src/relaxit/distributions/approx.py), also see [ℹ️ wiki](https://en.wikipedia.org/wiki/Logit-normal_distribution) and [💻 stackexchange](https://stats.stackexchange.com/questions/535560/approximating-the-logit-normal-by-dirichlet) 

## 🛠️ Install

### Install using pip
```bash
pip install relaxit
```

### Install from source
```bash
pip install git+https://github.com/intsystems/discrete-variables-relaxation
```

### Install via Git clone
```bash
git clone https://github.com/intsystems/discrete-variables-relaxation
cd discrete-variables-relaxation
pip install -e .
```

## 🚀 Quickstart 
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

## 🎮 Demo
| Laplace Bridge | REINFORCE in Acrobot environment | VAE with discrete latents |
| :---: | :---: | :---: |
| ![Laplace Bridge](https://github.com/user-attachments/assets/ac5d5a71-e7d7-4ec3-b9ca-9b72d958eb41) | ![REINFORCE](https://gymnasium.farama.org/_images/acrobot.gif) | ![VAE](https://github.com/user-attachments/assets/937585c4-df84-4ab0-a2b9-ea6a73997793) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intsystems/discrete-variables-relaxation/blob/main/demo/laplace-bridge.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intsystems/discrete-variables-relaxation/blob/main/demo/reinforce.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intsystems/discrete-variables-relaxation/blob/main/demo/demo.ipynb) |

For demonstration purposes, we divide our algorithms in three[^*] different groups. Each group relates to the particular demo code:
- [Laplace bridge between Dirichlet and LogisticNormal distributions](https://github.com/intsystems/discrete-variables-relaxation/blob/main/demo/laplace-bridge.ipynb)
- [REINFORCE](https://github.com/intsystems/discrete-variables-relaxation/blob/main/demo/reinforce.ipynb)
- [Other relaxation methods](https://github.com/intsystems/discrete-variables-relaxation/blob/main/demo/demo.ipynb)

We describe our demo experiments [here](https://github.com/intsystems/discrete-variables-relaxation/tree/main/demo).

[^*]: We also implement REINFORCE algorithm as a *score function* estimator alternative for our relaxation methods that are inherently *pathwise derivative* estimators. This one is implemented only for demo experiments and is not included into the source code of package.

## 📚 Stack
Some of the alternatives for GS were implemented in [pyro](https://docs.pyro.ai/en/dev/distributions.html), so we base our library on their codebase.
  
## 🧩 Some details
To make to library consistent, we integrate imports of distributions from `pyro` and `torch` into the library, so that all the categorical distributions can be imported from one entrypoint. 

## 👥 Contributors
- [Daniil Dorin](https://github.com/DorinDaniil) (Basic code writing, Final demo, Algorithms)
- [Igor Ignashin](https://github.com/ThunderstormXX) (Project wrapping, Documentation writing, Algorithms)
- [Nikita Kiselev](https://github.com/kisnikser) (Project planning, Blog post, Algorithms)
- [Andrey Veprikov](https://github.com/Vepricov) (Tests writing, Documentation writing, Algorithms)
- You are welcome to contribute to our project!

## 🔗 Useful links
- [About top-k GS](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html) 
- [VAE implementation with different latent distributions](https://github.com/kampta/pytorch-distributions)
- [KL divergence between Dirichlet and Logistic-Normal implemented in R](https://rdrr.io/cran/Compositional/src/R/kl.diri.normal.R)
- [About score function (SF) and pathwise derivate (PD) estimators, VAE and REINFORCE](https://arxiv.org/abs/1506.05254)
