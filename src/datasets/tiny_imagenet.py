import argparse
from pathlib import Path
import torch, torchvision
from torch.utils.data import DataLoader
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datasets.continual_dataset import ContinualDatasetInfo

class ContinualTinyImagenet:

    MEAN = (0.4802, 0.4481, 0.3975)
    STD = (0.2302, 0.2265, 0.2262)
    MAX_CLASSES = 200                # number of classes in the dataset

    def __init__(self, opt: argparse.Namespace):

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
                np.ones((opt.num_classes_per_task, opt.num_classes_per_task))
            )).to(opt.device) #Generates equal num_classes for all tasks. 

        assert(opt.num_tasks*opt.num_classes_per_task <= self.MAX_CLASSES), "num_classes lesser than classes_per_task * num_tasks"

        base_path = Path(opt.data_dir, 'TinyImagenet')

        if not base_path.exists():
            raise FileNotFoundError('TinyImagent was not found. See /utils/download_TinyImagenet.py')

        self.train_dataset = ImageFolder(
            root=Path(base_path, 'train'),  # type: ignore
            transform=self.train_transforms,
            target_transform=None
        )

        self.test_dataset = ImageFolder(
            root=Path(base_path, 'test'),  # type: ignore
            transform=self.test_transforms,
            target_transform=None
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
            num_classes=self.MAX_CLASSES,
            channels=3,
            size=(64, 64),
        )