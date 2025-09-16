import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.Tensor(self.data[idx])
        label = torch.Tensor([self.labels[idx]])
        return {'x': data, 'y': label}
