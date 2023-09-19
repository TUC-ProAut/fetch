from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path
from typing import Tuple
import torch
from compressors.compressor import CompressorDecompressorInfo
from encoders import EncoderInfo


def add_compressor_args(parser: ArgumentParser):
    parser.add_argument('--n_states', type=int, required=True, help='To how many states the input is quantized')
    parser.add_argument('--strategy', type=str, required=False, default='local', help='How to compute the quantiles. Local uses only the current tensor. tiny_imagenet_transfer uses precomputed values form the tiny_imagent dataset', choices=['local', 'tiny_imagenet_transfer', 'cifar10_transfer', 'cifar100_transfer'])


class Quantization:
    def __init__(self, opt: Namespace, encoder_info: EncoderInfo):
        """Quantizes the input to x states (ints are used).

        The tensor gets quantizised in --n_states equidistant quantiles. If --stategy is 'local' the quantiles are between the minimum and maximum value of the current input-featuremap.
        If --strategy is 'tiny_imagenet_transfer' the quantiles are based on the minimum and maximum values (outlier were removed) of tiny-imagenet.
        """
        self.in_size = encoder_info.out_size
        self.in_channels = encoder_info.out_channels
        self.compressed_size = encoder_info.out_size
        self.compressed_channels = encoder_info.out_channels
        self.out_size = encoder_info.out_size
        self.out_channels = encoder_info.out_channels
        
        assert encoder_info.out_size[0] != 0 and encoder_info.out_size[1] != 0
        assert encoder_info.out_channels != 0
        assert opt.n_states >= 2

        self.n_states = opt.n_states
        self.strategy = opt.strategy

        if self.strategy == 'tiny_imagenet_transfer':
            self.transfer_min, self.transfer_max = _get_statistics('TinyImagenet', encoder_info.name)
            self.transfer_delta = (self.transfer_max - self.transfer_min) / self.n_states
        elif self.strategy == 'cifar10_transfer':
            self.transfer_min, self.transfer_max = _get_statistics('CIFAR10_0_1', encoder_info.name)
            self.transfer_delta = (self.transfer_max - self.transfer_min) / self.n_states
        elif self.strategy == 'cifar100_transfer':
            self.transfer_min, self.transfer_max = _get_statistics('CIFAR100_0_49', encoder_info.name)
            self.transfer_delta = (self.transfer_max - self.transfer_min) / self.n_states


    def compress(self, x):
        if self.strategy == 'local':
            return self._compress_local(x)
        elif self.strategy == 'tiny_imagenet_transfer' or self.strategy == 'cifar10_transfer' or self.strategy == 'cifar100_transfer':
            return self._compress_transfer(x)
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')

    def _compress_transfer(self, x: torch.Tensor):
        out = torch.zeros_like(x) - 1
        out[x < self.transfer_min + self.transfer_delta] = self.transfer_min + 0.5 * self.transfer_delta
        for i_state in range(1, self.n_states-1):
            lower = self.transfer_min + i_state * self.transfer_delta
            upper = self.transfer_min + (i_state + 1) * self.transfer_delta
            mid = (lower + upper) / 2
            out[torch.logical_and(x >= lower, x < upper)] = mid
        out[x >= self.transfer_max - self.transfer_delta] = self.transfer_max - self.transfer_delta / 2
        return out
            
    def _compress_local(self, x):
        assert x.ndim >= 4

        orig_shape = x.shape
        out = torch.zeros_like(x) - 1
        x2 = x.view(x.size(0), -1)
        maxs, _ = x2.max(1)
        mins, _ = x2.min(1)
        deltas = (maxs - mins) / self.n_states
        # the dimension of maxs, mins, deltas corresponds to the images in the batch
        
        # TODO: vectorize
        for i_state in range(self.n_states):
            lower = mins + i_state * deltas
            upper = mins + (i_state+1) * deltas
            # reshape so we can comprate
            lower_t = lower[:, None, None, None].repeat(1, orig_shape[1], orig_shape[2], orig_shape[3])
            upper_t = upper[:, None, None, None].repeat(1, orig_shape[1], orig_shape[2], orig_shape[3])
            mid_t = (lower_t + upper_t) / 2

            if i_state == 0:
                mask = x<upper_t
            elif i_state == self.n_states-1:
                mask = x>=lower_t
            else:
                mask = torch.logical_and(x>=lower_t, x<upper_t)
            assert (out[mask] == -1).all().item()
            out[mask] = mid_t[mask]
        return out
    
    def decompress(self, x):
        return x

    def info(self):
        return CompressorDecompressorInfo(
            in_size=self.in_size,
            in_channels=self.in_channels,
            compressed_size=self.compressed_size,
            compressed_channels=self.compressed_channels,
            out_size=self.out_size,
            out_channels=self.out_channels,
        )


def _get_statistics(dataset: str, encoder:str) -> Tuple[float, float]:
    """get information about the maximum and  minimum values in the dataset.
    
    Args:
        dataset: name of the dataset
        encoder: name of the encoder (with parameters appended), use 'none' if there is no encoding
        
    Returns:
        (min, max): Min and Max of the Dataset with outliers removed
        
    Raises:
        ValueError: if there are no precomputed values for the (dataset, encoder)-pair
    """
    if (dataset, encoder) == ('TinyImagenet', 'passThrough'):
        return (-2.0860, 2.6636)
    elif (dataset, encoder) == ('TinyImagenet', 'cutr2') or (dataset, encoder) == ('TinyImagenet', 'cutr2.0'):
        return (0, 1.0502)
    elif (dataset, encoder) == ('TinyImagenet', 'cutr2.5'):
        return (0, 0.7106434106826782)
    elif (dataset, encoder) == ('TinyImagenet', 'cutr3') or (dataset, encoder) == ('TinyImagenet', 'cutr3.0'):
        return (0, 0.8490)
    elif (dataset, encoder) == ('TinyImagenet', 'cutr3.5'):
        return (0, 0.9108426570892334)
    elif (dataset, encoder) == ('TinyImagenet', 'cutr34_2'):
        return (0, 1.2152)
    elif (dataset, encoder) == ('TinyImagenet', 'cutr34_3'):
        return (0, 1.0785)
    elif (dataset, encoder) == ('CIFAR10_0_1', 'passThrough'):
        return (-2.2806, 2.7342)
    elif (dataset, encoder) == ('CIFAR10_0_1', 'cutr_cifar2') or (dataset, encoder) == ('CIFAR10_0_1', 'cutr_cifar2.0'):
        return (0, 3.2756)
    elif (dataset, encoder) == ('CIFAR10_0_1', 'cutr_cifar2.5'):
        return (0, 2.4295)
    elif (dataset, encoder) == ('CIFAR10_0_1', 'cutr_cifar3') or (dataset, encoder) == ('CIFAR10_0_1', 'cutr_cifar3.0'):
        return (0, 3.4424)
    elif (dataset, encoder) == ('CIFAR10_0_1', 'cutr_cifar3.5'):
        return (0, 2.6830)
    elif (dataset, encoder) == ('CIFAR100_0_49', 'passThrough'):
        return (2.7537312507629395, 2.596790313720703)
    elif (dataset, encoder) == ('CIFAR100_0_49', 'cutr32_1'):
        return (0, 3.0652577877044678)
    elif (dataset, encoder) == ('CIFAR100_0_49', 'cutr32_2'):
        return (0, 3.0054690837860107)
    elif (dataset, encoder) == ('CIFAR100_0_49', 'cutr32_3'):
        return (0, 4.810059547424316)
    else:
        raise ValueError(f'Cannot perform transfer-quantization-compression for dataset {dataset} and encoder {encoder}. First the values must be precomputed.')
