from argparse import ArgumentParser, Namespace
import random
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
from encoders import EncoderInfo


def add_sampler_args(parser: ArgumentParser) -> None:
    pass


class MemoryDataset(Dataset):
    def __init__(self, data_dict):
        """
        constructs a Dataset from a dictionary

        Args:
            data_dict: dictionary {label: [data]}
        """
        data = []
        targets = []
        for label, data_list in data_dict.items():
            for this_data in data_list:
                data.append(this_data)
                targets.append(label)

        assert len(data) == len(targets), "Data and Targets have wrong lengths"

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.targets[idx])


class GreedySampler:
    def __init__(self, opt: Namespace, encoder_info: EncoderInfo):
        """
        Greedy Sampler as described in 'GDumb: A Simple Approach that Questions Our Progress in Continual Learning'
        """

        self.num_slots = opt.memory_size
        self.mem = dict()    # maps the class to the examples

    def new_data(self, data: Any, target: int):
        """
        exposes the memory to a new data-label pair and stores it according to
        the greedy class-balanced sample algorithm
        
        Args:
            data: data
            target: label
        """
        target_was_seen = target in self.mem.keys()

        if not target_was_seen:
            # data needs to live on the cpu, see: https://discuss.pytorch.org/t/runtimeerror-cannot-re-initialize-cuda-in-forked-subprocess-to-use-cuda-with-multiprocessing-you-must-use-the-spawn-start-method/14083/2
            self.mem[target] = [torch.clone(data.to('cpu').detach())]
        else:
            n_slots_per_class = self.num_slots // len(self.mem.keys())
            class_has_empty_slots = len(self.mem.get(target, [])) < n_slots_per_class
            if class_has_empty_slots:
                self.mem[target].append(torch.clone(data.to('cpu').detach()))

        # While too full, remove from largest class
        class_sizes = {key: len(val) for key, val in self.mem.items()}
        mem_size = sum([val for val in class_sizes.values()])
        while mem_size > self.num_slots:
            largest_class = max(class_sizes, key=class_sizes.get) #type: ignore
            rand_idx = random.randint(0, len(self.mem[largest_class])-1)
            del self.mem[largest_class][rand_idx]
            class_sizes = {key: len(val) for key, val in self.mem.items()}
            mem_size = sum([val for val in class_sizes.values()])

    def new_batch(self, data: torch.Tensor, target: torch.Tensor) -> None:
        """
        exposes the memory to a new batch of size n and stores the data according to
        the greedy class-balanced sample algorithm
        
        Args:
            data: data-tensor of shape (n, ...)
            target: label-tensor of shape (n, 1)
        """
        bs = data.shape[0]
        for i in range(bs):
            self.new_data(data[i,...], int(target[i, ...].item()))
        

    def get_train_loader(self, opt: Namespace) -> DataLoader:
        """
        dataloader that iterates over the whole memory
        
        Args:
            opt: Experiment Configuration
            
        Returns
            Dataloader that iterates over the dadtaset"""
        ds = MemoryDataset(self.mem)
        return DataLoader(
            ds,
            shuffle=True,
            num_workers=opt.workers,
            batch_size=opt.batch_size,
            pin_memory=True                  # see https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/4
        )