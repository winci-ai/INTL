import torch
from torchvision.datasets import CIFAR10 as C10
from torchvision.datasets import CIFAR100 as C100
import torchvision.transforms as transforms
from abc import ABCMeta
from functools import lru_cache
from torch.utils.data import DataLoader

def get_data(model, loader, output_size, device):
    """ encodes whole dataset into embeddings """
    xs = torch.empty(
        len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            xs[i] = model(x).to(device)
            ys[i] = y.to(device)
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys

class BaseDataset(metaclass=ABCMeta):
    """
        base class for datasets, it includes 2 types:
            - for classifier training for evaluation,
            - for testing
    """
    def __init__(
        self, cfg, bs_clf=1000, bs_test=1000,
    ):
        self.bs_clf, self.bs_test = bs_clf, bs_test
        self.workers = cfg.workers
        self.dataset = cfg.dataset
        self.data_path = cfg.data_path
    
    @property
    @lru_cache()
    def clf(self):
        return DataLoader(
            dataset=self.ds_clf(),
            batch_size=self.bs_clf,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    @lru_cache()
    def test(self):
        return DataLoader(
            dataset=self.ds_test(),
            batch_size=self.bs_test,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )

class CIFAR10_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    def ds_clf(self):
        return C10(root=self.data_path, train=True, download=True, 
                transform=self.base_transform())

    def ds_test(self):
        return C10(root=self.data_path, train=False, download=True, 
                transform=self.base_transform())

class CIFAR100_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    def ds_clf(self):
        return C100(root=self.data_path, train=True, download=True, 
                    transform=self.base_transform())

    def ds_test(self):
        return C100(root=self.data_path, train=False, download=True, 
                    transform=self.base_transform())
