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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from relaxit.distributions import GaussianRelaxedBernoulli

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