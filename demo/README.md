# Demo experiments code
This repository contains our demo code for various experiments. The main demo code can be found in the notebook `demo.ipynb`. Open the notebook and run the cells to see the demonstration in action. For additional experiments, refer to the section [Additional experiments](#experiments). Before starting any experiments, ensure you follow all the installation steps outlined in the [Installation](#installation) section.

## Installation <a name="installation"></a>

To use this demo code, you need to have Python and the required packages installed on your computer.

```bash
# Clone the repository:
git clone https://github.com/intsystems/discrete-variables-relaxation.git

# Navigate to the repository directory:
cd discrete-variables-relaxation/demo

# Create Virtual Environment with Conda:
conda create --name relaxit-demo python=3.10

# Activate the conda environment:
conda activate relaxit-demo

# Install Dependencies
pip install -r requirements.txt
```
## Additional experiments<a name="experiments"></a>

For additional demo experiments, we have implemented Variational Autoencoders (VAEs) with different latent discrete distributions from **Just Relax It**. These implementations are adapted from the [PyTorch](https://github.com/pytorch/examples/tree/main/vae) examples repository.
1. **Train and save the models:**
   To run the additional demo code with VAEs, you need to train all the models and save their results. Execute the following commands:
    ```bash
    # VAE with Correlated Bernoulli latent space
    python vae_correlated_bernoulli.py

    # VAE with Gaussian Bernoulli latent space
    python vae_gaussian_bernoulli.py

    # VAE with Gumbel-Softmax top-K latent space
    python vae_gumbel_softmax_topk.py

    # VAE with Hard Concrete latent space
    python vae_hard_concrete.py
    
    # VAE with Invertible Gaussian latent space
    python vae_invertible_gaussian.py

    # VAE with Stochastic Times Smooth latent space
    python vae_stochastic_times_smooth.py

    # VAE with Straight Through Bernoullii latent space
    python vae_straight_through_bernoulli.py
    ```
2. **View the results:**
    After completing the training and testing of all the models, you can find the results of sampling and reconstruction methods in the directory `results`.

Moreover, we conducted experiments with Laplace Bridge between LogisticNormal and Dirichlet distributions. We use two-side Laplace bridge to approximate:
- Dirichlet using Logistic-Normal
- Logistic-Normal using Dirichlet
  
These experiments aim to find the best parameters to make the distributions nearly identical on the simplex. The experiments can be found in the notebook `laplace-bridge.ipynb`.

Furthermore, the Reinforce algorithm is applied in the [Acrobot environment](https://www.gymlibrary.dev/environments/classic_control/acrobot/). Detailed experiments can be viewed in the notebook `reinforce.ipynb`. A script `reinforce.py` can also be used for training.


