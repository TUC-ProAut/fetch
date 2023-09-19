# some general helper fuctions and classes

import torch
import numpy as np

def get_classwise_split_dict(targets):
    """
    Returns a dict that maps the targets to the indices in the whole dataset that have that target
    Arguments:
        indices (sequence): a sequence of indices
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict


class SubsetSequentialSampler(torch.utils.data.Sampler): #type: ignore
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)



class ReorderTargets(object):
    """
    Converts the class-orders to 0 -- (n-1) irrespective of order passed.
    """
    def __init__(self, class_order):
        self.class_order = np.array(class_order) 

    def __call__(self, target):
        return np.where(self.class_order==target)[0][0]