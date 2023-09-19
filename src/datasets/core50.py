import pathlib
import argparse
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import PIL.Image
from datasets.continual_dataset import ContinualDatasetInfo

class Core50Dataset(Dataset):
    def __init__(self, root: pathlib.Path, train: bool=True, transform=None, target_transform=None, reduced: bool = False):
        """Core50 Dataset
        
        Args:
            root_dir: Path to the dataset, must contain the directory 'core50_128x128'
            split: either 'train' or 'test'
            reduced: if set to true, images are sampleled at 1 fps from the video-stream
        """
        dataset_dir = root / 'core50_128x128'
        assert dataset_dir.is_dir()
        split = 'train' if train else 'test'
        index_file = dataset_dir / (split+'.csv')
        assert index_file.is_file(), 'Core50 must be preprocessed using /utils/preprocess_core50.py'
        self.df = pd.read_csv(index_file)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform

        if reduced:
            self.df = self.df.loc[self.df['frame'] % 20 == 0]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        assert (self.dataset_dir / record['file']).is_file, f"File not found: {self.dataset_dir / record['file']}"
        image = PIL.Image.open(self.dataset_dir / record['file'])
        label = record['class']
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label



class Core50:

    MEAN = (0.6002, 0.5722, 0.5418)
    STD = (0.1817, 0.1922, 0.2042)
    MAX_CLASSES = 10                # number of classes in the dataset

    def __init__(self, opt: argparse.Namespace, reduce_fps = False):
        data_dir = pathlib.Path(opt.data_dir)

        self.dataloader_kwargs = {
            'num_workers': opt.workers,
            'batch_size': opt.batch_size,
            'pin_memory': True
        }

        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.MEAN, self.STD),
        ])

        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.MEAN, self.STD),
        ])

        self.class_mask = torch.from_numpy(
            np.kron(
                np.eye(opt.num_tasks, dtype=int),
                np.ones((opt.num_classes_per_task, opt.num_classes_per_task), dtype=int)
            )).to(opt.device) #Generates equal num_classes for all tasks. 

        assert(opt.num_tasks*opt.num_classes_per_task <= self.MAX_CLASSES), "num_classes lesser than classes_per_task * num_tasks"

        self.train_dataset = Core50Dataset(
           train=True,
           transform=self.train_transforms,
           target_transform=None,
           root=data_dir,
           reduced=reduce_fps,
        )

        self.test_dataset = Core50Dataset(
            train=False,
            transform=self.test_transforms,
            target_transform=None,
            root=data_dir,
            reduced=reduce_fps,
        )

    def get_task_loaders(self) -> tuple[DataLoader, DataLoader]:

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
            size=(128, 128),
        )