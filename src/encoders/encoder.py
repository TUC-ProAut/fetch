from __future__ import annotations
from argparse import Namespace
from typing import Protocol, Tuple
from dataclasses import dataclass
from torch import Tensor
import torch.nn as nn
from datasets import ContinualDatasetInfo


@dataclass
class EncoderInfo():
    in_size: Tuple[int, int]
    in_channels: int
    out_size: Tuple[int, int]
    out_channels: int
    name: str

class Encoder(Protocol):
    def __init__(self, opts: Namespace, dataset_info: ContinualDatasetInfo):
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        ...

    def __call__(self, x) -> Tensor:
        ...

    def to(self, x) -> Encoder:
        ...

    def info(self) -> EncoderInfo:
        ...
