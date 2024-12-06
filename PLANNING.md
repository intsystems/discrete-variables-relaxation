# Project planning

In this file, we provide information about the planning of our work on the library named **Just Relax It** (`relaxit`). 

Since its implementation is carried out as a part of [BMM](https://github.com/intsystems/BMM) course, we consider it a full-fledged project named **Discrete variables relaxation** and therefore make detailed, long-term planning. 

This document is structured as follows:

1. [Motivation](#motivation)
2. [Algorithms to implement](#algorithms)
3. [Architecture of the project](#architecture)
4. [Schedule](#schedule)

## Motivation <a name="motivation"></a>

For lots of mathematical problems we need an ability to sample discrete random variables.
For instance, we may consider a VAE architecture with discrete latent space, e.g. Bernoulli or categorical.
The problem is that due to continuos nature of deep learning optimization, the usage of truely discrete random variables is infeasible. 
In particular, after sampling a variable from discrete distribution, we have not an ability to calculate the gradient through it.
Thus we use different relaxation methods.

## Algorithms to implement (from simplest to hardest) <a name="algorithms"></a>

In this project, we are going to implement the following algorithms:
1. [Relaxed Bernoulli](http://proceedings.mlr.press/v119/yamada20a/yamada20a.pdf)
2. [Correlated relaxed Bernoulli](https://openreview.net/pdf?id=oDFvtxzPOx)
3. [Gumbel-softmax TOP-K](https://arxiv.org/pdf/1903.06059)
4. [Straight-Through Bernoulli, distribution (don't mix with Relaxed distribution from pyro)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=62c76ca0b2790c34e85ba1cce09d47be317c7235)
5. [Invertible Gaussian reparametrization](https://arxiv.org/abs/1912.09588) with KL implemented
6. [Hard concrete](https://arxiv.org/pdf/1712.01312)
7. [REINFORCE](http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf)  (not a distribution actually, think how to integrate it with other distributions)
8. [Logit-normal distribution](https://en.wikipedia.org/wiki/Logit-normal_distribution) with KL implemented and [Laplace-form approximation of Dirichlet](https://stats.stackexchange.com/questions/535560/approximating-the-logit-normal-by-dirichlet)

You are invited to track our progress on the [main page](https://github.com/intsystems/relaxit/tree/main?tab=readme-ov-file#-algorithms-to-implement-from-simplest-to-hardest).

## Architecture of the project <a name="architecture"></a>

1. The most famous Python probabilistic libraries with a built-in differentiation engine are [PyTorch](https://pytorch.org/docs/stable/index.html) and [Pyro](https://docs.pyro.ai/en/dev/index.html). Specifically, we are mostly interested in the `distributions` package in both of them.
2. Base class for PyTorch-compatible distributions with Pyro support is `TorchDistribution`, for which we refer to [this page](https://docs.pyro.ai/en/dev/distributions.html#torchdistribution) on documentation. This should be the base class for almost all new Pyro distributions. Therefore in our project we are planning to inherit classes from this specific one.
3. To make our library compatible with modern deep learning packages, we will implement our classes with the following methods and properties, as it is mentioned in the Pyro documentation:
  > Derived classes must implement the methods `sample()` (or `rsample()` if `.has_rsample == True`) and `log_prob()`, and must implement the properties `batch_shape`, and `event_shape`. Discrete classes may also implement the `enumerate_support()` method to improve gradient estimates and set `.has_enumerate_support = True`.

```python
def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
    """
    Generates a sample_shape shaped reparameterized sample or sample_shape
    shaped batch of reparameterized samples if the distribution parameters
    are batched.
    """
    raise NotImplementedError
```

```python
def log_prob(self, value: torch.Tensor) -> torch.Tensor:
    """
    Returns the log of the probability density/mass function evaluated at
    `value`.
    
    Args:
        value (Tensor):
    """
    raise NotImplementedError
```

> [!NOTE]
> Below we present a diagram of the implementation of our project, demonstrating the class inheritance, as well as the methods necessary for implementation.

![Project scheme](assets/scheme.png)

## Schedule <a name="schedule"></a>

In order to getting all things done, we prepared a comprehensive schedule. 
We highlight the main events and deadlines that we are going meet to. 
All the contributions are assigned with their own tasks. 
Thus we suppose the project to be done in the distributed manner, exhibiting the best possible advantages from all the participants.

> [!NOTE]
> This version is preliminary, as the keypoints, i.e. techical meetings, have preliminary dates too.
> Moreover, up to date only main actions are noted.
> We will expand this schedule, providing a more detailed description of each task.

| Week # | By date | Deadline | Assignee | Task |
| :----: | :-----: | :------: | :------: | :--: |
| 1 | Oct 1  | TM 1 | Nikita | Repository, planning, presentation |
|   |        |      | Daniil, Igor, Andrey | Analyze papers, prepare info for slides |
| 2 | Oct 8  |      | Daniil | Think about basic code, create a distribution template (probably, use already implemented distribution with reparametrization like multivariate gaussian) |
|   |        |      | Igor   | Study repository structure templates, create necessary directories and files |
|   |        |      | Nikita | Think about a blogpost idea, check examples on the [habr.com](https://habr.com/) |
|   |        |      | Andrey | Study documentation types, make a list of advantages and disadvantages, propose the most convenient one |
| 3 | Oct 15 |      | Daniil | Think about code for demo (think, will it be VAE or not), understand how to expand it for all the algoritmhs |
|   |        |      | Igor   | Analyze how to make a documentation website using GitHub pages |
|   |        |      | Nikita | Make a list of the most important theory results to put into blogpost |
|   |        |      | Andrey | Understand how to test our code, check the main rules of the chosen documentation type |
| 4 | Oct 22 |      | Daniil | Make an intermediate version of basic code, including the example of usage: sampling and backpropagation |
|   |        |      | Igor   | Prepare a few pages of documentation via GitHub pages |
|   |        |      | Nikita | Create a plan and structure of the blogpost |
|   |        |      | Andrey | Find an example of creating a documentation of chosen type, make basic code drafty documentation, make some preliminary tests for basic code |
| 5 | Oct 29 | TM 2 | Daniil | Finalizing basic code |
|   |        |      | Nikita | Drafty version of blogpost, check this schedule for changes after TM 2 |
|   |        |      | Igor, Andrey | Drafty version of documentation |
| 6 | Nov 5  |      | Daniil | Choose the most convenient visualization for demo |
|   |        |      | Igor   | Extend the documentation with some of the algorithms |
|   |        |      | Nikita | Think about references for blogpost, help with documentation and demo |
|   |        |      | Andrey | Check tests for basic code and some algorithms |
| 7 | Nov 12 |      | Daniil | Prepare the demo and connect it with basic code and other algoritmhs |
|   |        |      | Igor   | Update the documentation, provide more detailed descriptions and examples |
|   |        |      | Nikita | Check the blogpost for typo and grammar, help with documentation and project wrapping |
|   |        |      | Andrey | Run tests on all the implemented algorithms, check documentation for correctness |
| 8 | Nov 19 | TM 3 | Igor   | Finalizing library, algoritmhs              |
|   |        |      | Daniil | Finalizing demo, algoritmhs                 |
|   |        |      | Andrey | Finalizing tests, documentation, algoritmhs |
|   |        |      | Nikita | Finalizing blogpost, algoritmhs             |

[^*]: Technical meeting
