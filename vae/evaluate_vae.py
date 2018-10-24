import os

import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
import yaml

from datasets import uci_dataloaders
import ssdkl


dataset_names = ['skillcraft', 'parkinsons', 'elevators', 'protein',
                 'blog', 'ctslice', 'buzz', 'electric']
# read data dir from config file
ssdkl_root = os.path.dirname(ssdkl.__file__)
with open(os.path.join(ssdkl_root, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
data_base = config['data_dir']

# Testing Dataset
dataset_dir = os.path.join(data_base, dataset_names[0])
N_max = 60000
print(dataset_dir)
trainloader, valloader = uci_dataloaders(dataset_dir, N_max=N_max)


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

    def reparameterize(self, mu, logvar):
        if self.training:
            # Logvar
            std = logvar.mul(0.5).exp_()
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


input_dims = [18, 20, 18, 9, 280, 384, 77, 6]
results_base = 'vae_checkpoints'
new_results_base = 'vae_results'
z_dims = [2]
folds = list(range(10))
labeleds = [100, 300]
count = 0
for idx, dataset in enumerate(dataset_names):
    print('Dataset: {}'.format(dataset))
    input_dim = input_dims[idx]
    data_dir = os.path.join(data_base, dataset)
    y = np.load(os.path.join(data_dir, 'y.npy'))
    feat_dir = os.path.join(new_results_base, dataset)
    os.makedirs(feat_dir)
    np.save(os.path.join(feat_dir, 'y.npy'), y)
    print('Data dir: {}'.format(data_dir))
    for z_dim in z_dims:
        print('Z dim: {}'.format(z_dim))
        # Load model
        results_dir = os.path.join(results_base, '{}_{}'.format(dataset, z_dim))
        model = VAE(input_dim, z_dim=z_dim)
        model.cuda()
        model.load_state_dict(torch.load(os.path.join(results_dir, 'best.ckpt')))
        print('Loaded model')

        # Get features
        X_in = np.load(os.path.join(data_dir, 'X.npy'))
        x = torch.from_numpy(X_in).float()
        data = Variable(x).cuda()
        _, z, _ = model(data)
        z = z.data.cpu().numpy()

        # Save features
        feat_fn = os.path.join(feat_dir, 'X{}.npy'.format(z_dim))
        np.save(feat_fn, z)
        print('Saved features')

# ## Evaluating MLP performance

z_dims = [2]
labeleds = [100, 300]
n_test = 1000
trials = 10

results = {}
for dataset in dataset_names:
    print('Dataset: {}'.format(dataset))
    results_dir = os.path.join(new_results_base, dataset)
    for z_dim in z_dims:
        print(' Z dim: {}'.format(z_dim))
        X_in = np.load(os.path.join(results_dir, 'X{}.npy'.format(z_dim)))
        y_in = np.load(os.path.join(results_dir, 'y.npy'))
        # Shuffle X and y, then select train and test
        for labeled in labeleds:
            print('  Labels: {}'.format(labeled))
            trial_results = np.zeros((trials,))
            for trial in range(trials):
                X, y = shuffle(X_in, y_in)
                X_tr, X_te = X[:labeled], X[labeled:n_test+labeled]
                y_tr, y_te = y[:labeled], y[labeled:n_test+labeled]

                # Train MLP
                mlp = MLPRegressor(hidden_layer_sizes=(20,),
                    early_stopping=True, validation_fraction=0.1, max_iter=5000)
                mlp.fit(X_tr, y_tr)
                y_pred = mlp.predict(X_te)
                rmse = np.sqrt(np.mean((y_pred - y_te) ** 2))
                trial_results[trial] = rmse
            results[(dataset, z_dim, labeled)] = trial_results

print(results)
