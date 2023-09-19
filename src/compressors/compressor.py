from typing import Protocol, Tuple
from dataclasses import dataclass
from argparse import Namespace
import torch
from encoders import EncoderInfo
from encoders.encoder import Encoder

@dataclass
class CompressorDecompressorInfo:
    in_size: Tuple[int, int]
    in_channels: int
    compressed_size: Tuple[int, int]
    compressed_channels: int
    out_size: Tuple[int, int]
    out_channels: int


class CompressorDecompressor(Protocol):
    def __init__(self, opt: Namespace, encoder_info: EncoderInfo):
        ...
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def info(self) -> CompressorDecompressorInfo:
        ...