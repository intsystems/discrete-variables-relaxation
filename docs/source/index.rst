:github_url: https://github.com/intsystems/relaxit

Just Relax It
=============

.. image:: _static/img/logo.png
  :width: 200
  :align: center

.. raw:: html

   <div style="margin-bottom: 20px;"></div>

"Just Relax It" is a cutting-edge Python library designed to streamline the optimization of discrete probability distributions in neural networks, offering a suite of advanced relaxation techniques compatible with PyTorch.

Motivation
----------

For lots of mathematical problems we need an ability to sample discrete random variables.
The problem is that due to continuous nature of deep learning optimization, the usage of truly discrete random variables is infeasible.
Thus we use different relaxation methods.
One of them, `Concrete distribution <https://arxiv.org/abs/1611.00712>`_ or `Gumbel-Softmax <https://arxiv.org/abs/1611.01144>`_ (this is one distribution proposed in parallel by two research groups) is implemented in different DL packages.
In this project we implement different alternatives to it.

.. image:: _static/img/overview.png
  :width: 600
  :align: center

.. raw:: html

   <div style="margin-bottom: 20px;"></div>

.. toctree::
   :maxdepth: 1
   :caption: Guidelines

   install
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Code

   relaxit.distributions
