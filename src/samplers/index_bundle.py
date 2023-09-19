from __future__ import annotations
from argparse import ArgumentParser, Namespace
import random
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
from encoders import EncoderInfo


def add_sampler_args(parser: ArgumentParser) -> None:
    parser.add_argument('--bundle_size', type=int, required=True, help='Number of Samples that are bundeled together')


class _IndexBundleDataset(Dataset):
    def __init__(self, data_dict, bundle_idx_vecs):
        """Constructs a dataset from an dictionary of bundles
        
        Args:
            data_dict: dictionary that maps targets to list of data
            bundle_idx_vecs: the index-vectors used ot query the bundle
        """
        self.data = []
        self.targets = []
        self.bundle_size = len(bundle_idx_vecs)
        for target, data_list in data_dict.items():
            for this_bundle in data_list:
                for this_idx_vec in bundle_idx_vecs[:this_bundle.size]:
                    extracted_vec = this_bundle.unbind(this_idx_vec).data
                    self.data.append(extracted_vec)
                    self.targets.append(target)
        assert len(self.data)==len(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self ,idx):
        return self.data[idx], self.targets[idx]


class HdVector:
    """Hyperdimensional Vector for the MAP-Architecture
    
    Args:
        data: the content of the vector. Must have a single dimension
        size (optional): the number of bundeled vectors
        
    Attributes:
        data (torch.Tensor): Data of the vector
        size: number of bundeled vectors
    """
    def __init__(self, data: torch.Tensor, size: int=1):
        self.size = size       # number of bundeled HdVectors
        if data.dim() != 1:
            raise ValueError('the tensor needs to have only one dimension')
        else:
            self.data = data

    def bind(self, other: HdVector) -> HdVector:
        """elementwise multiplication
        """
        return HdVector(self.data * other.data)

    def unbind(self, other: HdVector) -> HdVector:
        """elementwise multiplication
        """
        return HdVector(self.data * other.data)

    def bundle(self, other: HdVector) -> HdVector:
        """elementwise addition
        """
        return HdVector(self.data + other.data, size=self.size+other.size)

    def sim(self, other: HdVector) -> float:
        """cosine similarity
        """
        dot = torch.dot(self.data, other.data)
        abs = self.data.norm() * other.data.norm()
        return (dot / abs).item()


def _random_hd_vec(dim: int) -> HdVector:
    """generates random bipolar hyperdimensional vector of size dim
    
    Args:
        dim: size of the input vector
        
    Returns: 
        The hyperdimentional vector"""
    x = torch.randint(0, 2, (dim,))
    x[x==0] = -1
    return HdVector(x)


class IndexBundle:
    """Greedy Sampling strategy with compression. Here each memory-slot is occupied by a bundle
    The bundle b is computed as

    \displaystyle \mathbf b^{(t)}_j := \bigoplus_{i=0}^{N-1} \mathbf x^{(t)}_{j+i} \otimes \mathbf i_i

    where x is the datavector and i is a bipolar index-vector. To get back the data, an unbinding-
    operation is applied to the bundle using the index-vector.
    """
    def __init__(self, opt: Namespace, encoder_info: EncoderInfo):
        self.bundle_size = opt.bundle_size
        self.hdv_size = encoder_info.out_channels
        self.num_slots = opt.memory_size
        self.mem = dict()    # maps the class to the examples

        if encoder_info.out_size != (0, 0):
            raise ValueError(f'Outputs of Encoder must have a size of (0,0), got {encoder_info.out_size}')

        self.bundle_idx_vecs = [_random_hd_vec(self.hdv_size) for _ in range(self.bundle_size)]

    def _bundle_at_end(self, data: torch.Tensor, target: int) -> None:
        """appends the new vector at the end of the memory. If the bundle in the last
        slot is smaller than specified, the new data gets bundled.
        Otherwise the new data gets appended at the end.
        """
        if len(self.mem[target]) == 0:
            make_new_bundle = True
        else:
            tail_bundle_sz = self.mem[target][-1].size
            if tail_bundle_sz < self.bundle_size:
                make_new_bundle = False
            else:
                make_new_bundle = True

        data_vec = HdVector(data)

        if make_new_bundle:
            idx_vec = self.bundle_idx_vecs[0]
            self.mem[target].append(data_vec.bind(idx_vec))
        else:
            idx = self.mem[target][-1].size
            idx_vec = self.bundle_idx_vecs[idx]
            binding = data_vec.bind(idx_vec)
            bundle = self.mem[target][-1].bundle(binding)
            self.mem[target][-1] = bundle
            
    def new_data(self, data: Any, target: int) -> None:
        target_was_seen = target in self.mem.keys()

        if not target_was_seen:
            # data needs to live on the cpu, see: https://discuss.pytorch.org/t/runtimeerror-cannot-re-initialize-cuda-in-forked-subprocess-to-use-cuda-with-multiprocessing-you-must-use-the-spawn-start-method/14083/2
            self.mem[target] = []
            self._bundle_at_end(torch.clone(data.to('cpu').detach()), target)
        else:
            n_slots_per_class = self.num_slots // len(self.mem.keys())
            class_has_empty_slots = len(self.mem.get(target, [])) < n_slots_per_class
            if class_has_empty_slots:
                self._bundle_at_end(torch.clone(data.to('cpu').detach()), target)

        self.trim_memory()

    def trim_memory(self) -> None:
        """while too full, remove from largest class
        """
        class_sizes = {key: len(val) for key, val in self.mem.items()}
        mem_size = sum([val for val in class_sizes.values()])
        while mem_size > self.num_slots:
            largest_class = max(class_sizes, key=class_sizes.get) #type: ignore
            rand_idx = random.randint(0, len(self.mem[largest_class])-1)
            del self.mem[largest_class][rand_idx]
            class_sizes = {key: len(val) for key, val in self.mem.items()}
            mem_size = sum([val for val in class_sizes.values()])

    def new_batch(self, data: torch.Tensor, target: torch.Tensor) -> None:
        bs = data.shape[0]
        for i in range(bs):
            self.new_data(data[i,...], int(target[i, ...].item()))
        

    def get_train_loader(self, opt: Namespace) -> DataLoader:
        """dataloader that extracts data-samples out of the bundles
        
        Args:
            opt: Experiment Configuration
            
        Returns:
            Dataloader that iterates over the dadtaset
        """
        ds = _IndexBundleDataset(self.mem, self.bundle_idx_vecs)
        return DataLoader(
            ds,
            shuffle=True,
            num_workers=opt.workers,
            batch_size=opt.batch_size,
            pin_memory=True                  # see https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/4
        )