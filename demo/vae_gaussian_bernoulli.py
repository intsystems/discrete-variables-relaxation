import os
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.distributions as td
import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

class GaussianRelaxedBernoulli(TorchDistribution):
    """
    Gaussian-based continuous Relaxed Bernoulli distribution class inheriting from Pyro's TorchDistribution.

    Parameters:
    - loc (Tensor): The mean (mu) of the normal distribution.
    - scale (Tensor): The standard deviation (sigma) of the normal distribution.
    """

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args: bool = None):
        """
        Initializes the GaussianRelaxedBernoulli distribution.

        Args:
        - loc (Tensor): Mean of the normal distribution.
        - scale (Tensor): Standard deviation of the normal distribution.
        - validate_args (bool): Whether to validate arguments.
        """
        self.loc = loc.float()  # Ensure loc is a float tensor
        self.scale = scale.float()  # Ensure scale is a float tensor
        self.normal = torch.distributions.Normal(0, self.scale)
        super().__init__(validate_args=validate_args)

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape of the distribution.

        The batch shape represents the shape of independent distributions.
        For example, if `loc` is vector of length 3,
        the batch shape will be `[3]`, indicating 3 independent Bernoulli distributions.
        """
        return self.loc.shape

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.

        The event shape represents the shape of each individual event.
        """
        return torch.Size()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample from the distribution using the reparameterization trick.

        Args:
        - sample_shape (torch.Size): The shape of the sample.

        Returns:
        - torch.Tensor: A sample from the distribution.
        """
        eps = self.normal.sample(sample_shape)
        print(eps)
        z = torch.clamp(self.loc + eps, 0, 1)
        # probs = z
        return z

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample from the distribution.

        Args:
        - sample_shape (torch.Size): The shape of the sample.

        Returns:
        - torch.Tensor: A sample from the distribution.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs('./results/vae_gaussian_bernoulli', exist_ok=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

INITIAL_TEMP = 1.0
ANNEAL_RATE = 0.00003
MIN_TEMP = 0.1

temp = INITIAL_TEMP
steps = 0


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, temp=1.0, hard=False):
        mu = self.encode(x.view(-1, 784))
        q_z = GaussianRelaxedBernoulli(mu, torch.tensor([1], device=device))
        z = q_z.rsample()  # sample with reparameterization

        if hard:
            # No step function in torch, so using sign instead
            z_hard = 0.5 * (torch.sign(z) + 1)
            z = z + (z_hard - z).detach()

        return self.decode(z), z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, prior=0.5, eps=1e-10):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # You can also compute p(x|z) as below, for binary output it reduces
    # to binary cross entropy error, for gaussian output it reduces to
    t1 = mu * ((mu + eps) / prior).log()
    t2 = (1 - mu) * ((1 - mu + eps) / (1 - prior)).log()
    KLD = torch.sum(t1 + t2, dim=-1).sum()

    return BCE + KLD


def train(epoch):
    global temp, steps
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, q_z = model(data, temp=temp)
        loss = loss_function(recon_batch, data, q_z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        steps += 1
        if steps % 1000 == 0:
            temp = max(temp * np.exp(-ANNEAL_RATE * steps), MIN_TEMP)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    global temp
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, q_z = model(data, temp=temp)
            test_loss += loss_function(recon_batch, data, q_z).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/vae_gaussian_bernoulli/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = np.random.binomial(1, 0.5, size=(64, 20))
            sample = torch.from_numpy(np.float32(sample)).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/vae_gaussian_bernoulli/sample_' + str(epoch) + '.png')