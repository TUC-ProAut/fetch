from argparse import Namespace
from typing import Protocol, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np

@dataclass
class ContinualDatasetInfo:
    num_classes: int
    channels: int
    size: Tuple[int, int]

class ContinualDataset(Protocol):
    class_mask: np.ndarray

    def __init__(self, opt: Namespace):
        ...

    def get_task_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """returns (train_loader, test_loader)"""
        ...

    def info(self) -> ContinualDatasetInfo:
        ...