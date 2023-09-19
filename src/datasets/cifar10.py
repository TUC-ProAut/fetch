import argparse
from pathlib import Path
import torch, torchvision
from torch.utils.data import DataLoader
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from datasets.continual_dataset import ContinualDatasetInfo

class ContinualCifar10:

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    MAX_CLASSES = 10                # number of classes in the dataset

    def __init__(self, opt: argparse.Namespace):
        data_dir = Path(opt.data_dir)

        self.dataloader_kwargs = {
            'num_workers': opt.workers,
            'batch_size': opt.batch_size,
            'pin_memory': True}

        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.MEAN, self.STD)])

        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.MEAN, self.STD)])

        self.class_mask = torch.from_numpy(
            np.kron(
                np.eye(opt.num_tasks, dtype=int),
                np.ones((opt.num_classes_per_task, opt.num_classes_per_task), dtype=int)
            )).to(opt.device) #Generates equal num_classes for all tasks. 

        assert(opt.num_tasks*opt.num_classes_per_task <= self.MAX_CLASSES), "num_classes lesser than classes_per_task * num_tasks"

        self.train_dataset = CIFAR10(
           train=True,
           transform=self.train_transforms,
           target_transform=None,
           root=data_dir,
           download=True
        )

        self.test_dataset = CIFAR10(
            train=False,
            transform=self.test_transforms,
            target_transform=None,
            root=data_dir,
            download=True
        )

    def get_task_loaders(self) -> Tuple[DataLoader, DataLoader]:

        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.dataloader_kwargs
        )

        test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            **self.dataloader_kwargs
        )

        return train_loader, test_loader

    def info(self) -> ContinualDatasetInfo:
        return ContinualDatasetInfo(
            num_classes=10,
            channels=3,
            size=(32, 32),
        )