import os
import argparse
import numpy as np
import torch
import sys
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from relaxit.distributions import CorrelatedRelaxedBernoulli


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="VAE MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    return parser.parse_args()


args = parse_arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

os.makedirs("./results/vae_correlated_bernoulli", exist_ok=True)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

steps = 0


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with Correlated Relaxed Bernoulli distribution.
    """

    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        # Initialize R as some symmetric positive definite matrix
        self.R = torch.eye(20, device=device)
        self.tau = torch.tensor(0.1, device=device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input by passing through the encoder network
        and return the latent code.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent code.
        """
        h1 = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent code by passing through the decoder network
        and return the reconstructed input.

        Args:
            z (torch.Tensor): Latent code.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Reconstructed input and latent code.
        """
        pi = self.encode(x.view(-1, 784))
        pi = torch.clamp(pi, min=1e-6, max=1 - 1e-6)
        q_z = CorrelatedRelaxedBernoulli(pi, self.R, self.tau)
        z = q_z.rsample()  # sample with reparameterization

        return self.decode(z), pi


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    q_z: torch.Tensor,
    prior: float = 0.5,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute the loss function for the VAE.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        q_z (torch.Tensor): Latent code.
        prior (float): Prior probability.
        eps (float): Small value to avoid log(0).

    Returns:
        torch.Tensor: Loss value.
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    t1 = q_z * ((q_z + eps) / prior).log()
    t2 = (1 - q_z) * ((1 - q_z + eps) / (1 - prior)).log()
    KLD = torch.sum(t1 + t2, dim=-1).sum()

    return BCE + KLD


def train(epoch: int) -> None:
    """
    Train the VAE for one epoch.

    Args:
        epoch (int): Current epoch number.
    """
    global steps
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, q_z = model(data)
        loss = loss_function(recon_batch, data, q_z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

        steps += 1

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(epoch: int) -> None:
    """
    Test the VAE for one epoch.

    Args:
        epoch (int): Current epoch number.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, q_z = model(data)
            test_loss += loss_function(recon_batch, data, q_z).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    "results/vae_correlated_bernoulli/reconstruction_"
                    + str(epoch)
                    + ".png",
                    nrow=n,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = np.random.binomial(1, 0.5, size=(64, 20))
            sample = torch.from_numpy(np.float32(sample)).to(device)
            sample = model.decode(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                "results/vae_correlated_bernoulli/sample_" + str(epoch) + ".png",
            )
