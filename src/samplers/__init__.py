from argparse import Namespace
from samplers.sampler import Sampler
import samplers.greedy_sampler
import samplers.index_bundle
from encoders import EncoderInfo

ALL_SAMPLERS = ['greedy_sampler', 'index_bundle']


def get_sampler(opts: Namespace, encoder_info: EncoderInfo) -> Sampler:
    """
    returns the sampler object

    Args
        opts: experiment options
        dataset: dataset object (only to read some information)
    """
    if opts.sampler == 'greedy_sampler':
        return samplers.greedy_sampler.GreedySampler(opts, encoder_info)  # type: ignore
    elif opts.sampler == 'index_bundle':
        return samplers.index_bundle.IndexBundle(opts, encoder_info)  # type: ignore
    else:
        raise ValueError(f'{opts.sampler} is not a valid sampler')


def get_sampler_arg_fn(sampler_name: str):
    """
    returns a handle to a funciton that adds encoder-specifig arguments to the argparser
    
    Args:
        sampler_name: name of the sampler
    """
    if sampler_name == 'greedy_sampler':
        return samplers.greedy_sampler.add_sampler_args
    elif sampler_name == 'index_bundle':
        return samplers.index_bundle.add_sampler_args
    else:
        raise ValueError(f'{sampler_name} is not a valid sampler')
    