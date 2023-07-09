from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch import tensor
import numpy as np

class PeptideData(Dataset):
    def __init__(self, X, labels, masks, device):
        super(PeptideData, self).__init__()

        self.X = X
        self.y = labels
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return tensor(self.X[index], dtype=torch.float32, device=self.device), \
                tensor(self.y[index], dtype=torch.int, device=self.device), \
                tensor(self.masks[index], dtype=torch.bool, device=self.device)

class LabelEmbeddingData(PeptideData):
    def __init__(self, X, labels, masks, device):
        super().__init__(X, labels, masks, device)

        self.label_input = np.repeat(np.array([range(0, 15)]), self.y.shape[0], axis=0)

    def __getitem__(self, index):
        return tensor(self.X[index], dtype=torch.float32, device=self.device), \
                tensor(self.y[index], dtype=torch.int, device=self.device), \
                tensor(self.masks[index], dtype=torch.bool, device=self.device), \
                tensor(self.label_input[index], dtype=torch.long, device=self.device)

class BalancedData(Dataset):
    def __init__(self, X, labels, masks, device):
        super(BalancedData, self).__init__()

        self.X = X
        self.y = labels
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return tensor(self.X[index], dtype=torch.float32, device=self.device), \
                tensor(self.y[index], dtype=torch.float32, device=self.device), \
                tensor(self.masks[index], dtype=torch.bool, device=self.device)



class ImbalancedMultilabelDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        dataset: dataset tp resample
        labels: one-hot labels
        num_samples: number of samples to generate
    """

    def __init__(self, dataset, labels: np.array, num_samples: int = None):

        #  all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        self.num_samples = 2 * len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        n, m = labels.shape # n samples, n_class

        weights_per_label = 1.0 / np.sum(labels, axis=0)
        weights_per_sample = []

        for i in range(n):
            w = np.sum(weights_per_label[labels[i, :] == 1])
            weights_per_sample.append(w)

        self.weights = torch.DoubleTensor(weights_per_sample)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples