from typing import Protocol, Any
from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader


class Sampler(Protocol):
    def __init__(self, opt: Namespace):
        ...

    def new_data(self, data: Any, target: int):
        """
        exposes the memory to a new data-label pair and stores it
        in the memory
        
        Args:
            data: data
            target: label
        """
        ...

    def new_batch(self, data: torch.Tensor, target: torch.Tensor) -> None:
        """
        exposes the memory to a new batch of size n and stores each
        data-label pair in the memory
        
        Args:
            data: data-tensor of shape (n, ...)
            target: label-tensor of shape (n, 1)
        """
        ...
        

    def get_train_loader(self, opt: Namespace) -> DataLoader:
        """
        dataloader that iterates over the whole memory
        
        Args:
            opt: Experiment Configuration
            
        Returns:
            Dataloader that iterates over the dadtaset"""
        ...