from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import os


class UCIDataset(Dataset):
    def __init__(self, dataset_dir, idxs=None, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.X = np.load(os.path.join(self.dataset_dir, 'X.npy'))
        if idxs is None: idxs = np.arange(len(self.X))
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        x = self.X[i]
        x = np.expand_dims(x, axis=0)
        if self.transform:
            x = self.transform(x)
        return x


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, x):
        x = torch.from_numpy(x).float()
        return x


def uci_dataloader(dataset_dir, idxs=None, batch_size=64, shuffle=True,
    num_workers=4):
    transform_list = []
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = UCIDataset(dataset_dir, idxs=idxs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader


def uci_dataloaders(dataset_dir, N_max=None, p_val=0.1, batch_size=64,
    shuffle=True, num_workers=4):
    idxs = np.arange(len(np.load(os.path.join(dataset_dir, 'X.npy'))))
    np.random.shuffle(idxs)
    if N_max is not None and N_max < len(idxs):
        idxs = idxs[:N_max]
    n_val = int(p_val * len(idxs))
    val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
    trainloader = uci_dataloader(dataset_dir, idxs=train_idxs,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valloader = uci_dataloader(dataset_dir, idxs=val_idxs,
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return (trainloader, valloader)