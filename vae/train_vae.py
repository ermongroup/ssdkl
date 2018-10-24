from __future__ import print_function
import os
import shutil
import time

import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import yaml

from datasets import uci_dataloaders
import ssdkl


parser = argparse.ArgumentParser(description='VAE for UCI datasets')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
# Added stuff
parser.add_argument('--N_max', type=int, default=60000,
    help='Maximum number of unlabeled training examples')
parser.add_argument('--gpu', type=int, default=0,
    help='Index of GPU to use')
parser.add_argument('--dataset', type=str, default='skillcraft',
    help='Name of UCI dataset to use')
parser.add_argument('--z_dim', type=int, default=2,
    help='Dimension of latent vector')
parser.add_argument('--verbose', action='store_true', default=False,
    help='Print losses at each epoch')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)

args.cuda = not args.no_cuda and torch.cuda.is_available()
time.sleep(1.0)
print('Using GPU {}'.format(args.gpu))


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# read data dir from config file
ssdkl_root = os.path.dirname(ssdkl.__file__)
with open(os.path.join(ssdkl_root, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

dataset_dir = os.path.join(config['data_dir'], args.dataset)
results_dir = os.path.join('vae_checkpoints', '{}_{}'.format(args.dataset, args.z_dim))
os.makedirs(results_dir)

X = np.load(os.path.join(dataset_dir, 'X.npy'))
input_dim = X.shape[1]  # need to get this (dimensionality) for each dataset

# Make train and val loader
trainloader, valloader = uci_dataloaders(dataset_dir, N_max=60000,
    batch_size=args.batch_size)


class VAE(nn.Module):
    def __init__(self, input_dim, z_dim=2):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.input_dim, 100)
        self.fc1_2 = nn.Linear(100, 50)
        self.fc1_3 = nn.Linear(50, 50)
        self.fc21 = nn.Linear(50, self.z_dim)
        self.fc22 = nn.Linear(50, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 50)
        self.fc3_1 = nn.Linear(50, 50)
        self.fc3_2 = nn.Linear(50, 100)
        self.fc4 = nn.Linear(100, self.input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc1_2(h1))
        h3 = self.relu(self.fc1_3(h2))
        return self.fc21(h3), self.fc22(h3)
        # return self.fc21(h1), self.relu(self.fc22(h1))

    def reparameterize(self, mu, logvar):
        if self.training:
            # Logvar
            std = logvar.mul(0.5).exp_()
            # Var
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc3_1(h3))
        h5 = self.relu(self.fc3_2(h4))
        return self.sigmoid(self.fc4(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        REC = torch.mean(
            ((recon_x - x.view(-1, self.input_dim)) ** 2).sum(dim=1))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        # Logvar
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1))
        loss = REC + KLD
        return loss, REC, KLD


model = VAE(input_dim, z_dim=args.z_dim)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    train_loss_rec = 0
    train_loss_kld = 0
    for batch_idx, data in enumerate(trainloader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss, loss_rec, loss_kld = model.loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.data.item()
        train_loss_rec += loss_rec.data.item()
        train_loss_kld += loss_kld.data.item()
        optimizer.step()

    avg_loss = train_loss / len(trainloader.dataset)
    avg_rec = train_loss_rec / len(trainloader.dataset)
    avg_kl = train_loss_kld / len(trainloader.dataset)

    if args.verbose:
        print('====> Epoch: {} Avg train loss: {:.4f}, rec {:.4f}, kl {:.4f}'.format(epoch, avg_loss, avg_rec, avg_kl))

    return avg_loss, avg_rec, avg_kl


def val(epoch):
    model.eval()
    val_loss = 0
    val_loss_rec = 0
    val_loss_kld = 0
    for i, data in enumerate(valloader):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        recon_batch, mu, logvar = model(data)
        loss, loss_rec, loss_kld = model.loss_function(recon_batch, data, mu, logvar)
        val_loss += loss.data.item()
        val_loss_rec += loss_rec.data.item()
        val_loss_kld += loss_kld.data.item()

    avg_loss = val_loss / len(valloader.dataset)
    avg_rec = val_loss_rec / len(valloader.dataset)
    avg_kl = val_loss_kld / len(valloader.dataset)

    if args.verbose:
        print('====> Val set loss: {:.4f}, rec {:.4f}, kl {:.4f}\n'.format(
            avg_loss, avg_rec, avg_kl))

    return avg_loss, avg_rec, avg_kl


train_losses = np.zeros((args.epochs, 3))
val_losses = np.zeros((args.epochs, 3))

# Delete results dir if existing and make new one
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
    # raise Exception('Results directory exists')
os.makedirs(results_dir)

best_val = 1e10
best_model_fn = None
for epoch in range(1, args.epochs + 1):
    avg_loss_train, avg_rec_train, avg_kl_train = train(epoch)
    avg_loss_val, avg_rec_val, avg_kl_val = val(epoch)
    # Store losses
    train_losses[epoch-1, 0] = avg_loss_train
    train_losses[epoch-1, 1] = avg_rec_train
    train_losses[epoch-1, 2] = avg_kl_train
    val_losses[epoch-1, 0] = avg_loss_val
    val_losses[epoch-1, 1] = avg_rec_val
    val_losses[epoch-1, 2] = avg_kl_val
    # Save best model
    if avg_loss_val < best_val:
        if best_model_fn is not None:
            os.remove(best_model_fn)
        best_model_fn = os.path.join(results_dir, 'epoch{}.ckpt'.format(epoch))
        torch.save(model.state_dict(), best_model_fn)
        best_val = avg_loss_val
        if args.verbose:
            print('  New best! Saved model!\n')

# Save losses
np.save(os.path.join(results_dir, 'train_losses.npy'), train_losses)
np.save(os.path.join(results_dir, 'val_losses.npy'), val_losses)

# Copy best model to standard name
shutil.copy(best_model_fn, os.path.join(results_dir, 'best.ckpt'))
