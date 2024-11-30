:github_url: https://github.com/intsystems/discrete-variables-relaxation

Just Relax It
=============

.. image:: ../assets/logo.png
  :width: 200
  :align: center

"Just Relax It" is a cutting-edge Python library designed to streamline the optimization of discrete probability distributions in neural networks, offering a suite of advanced relaxation techniques compatible with PyTorch.

Motivation
----------

For lots of mathematical problems we need an ability to sample discrete random variables. 
The problem is that due to continuos nature of deep learning optimization, the usage of truely discrete random variables is infeasible. 
Thus we use different relaxation method. 
One of them, `Concrete distribution <https://arxiv.org/abs/1611.00712>`_ or `Gumbel-Softmax <https://arxiv.org/abs/1611.01144>`_ (this is one distribution proposed in parallel by two research groups) is implemented in different DL packages. 
In this project we implement different alternatives to it. 

.. image:: ../assets/overview.png
  :width: 600
  :align: center

.. toctree::
   :maxdepth: 1
   :caption: Guidelines
   
   install
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Code

   source/modules
   source/relaxit.distributions

