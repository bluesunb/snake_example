import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from typing import Union, List, Optional, Type, Dict, Any, Tuple
from stable_baselines3.common.type_aliases import Schedule


def create_mlp(input_dim: int,
               output_dim: int,
               net_arch: List[Union[int, str]],
               activation_fn: Type[nn.Module],
               output_activation: Optional[nn.Module] = None,
               net_kwargs: Optional[dict] = None) -> List[nn.Module]:
    """
    Creates a multi-layer perceptron (MLP) with `n_layers` hidden layers, each of size `size`.
    The network takes inputs of dimension `input_dim` and returns outputs of dimension `output_dim`.
    The network uses `activation` as the activation function for the hidden layers, and
    `output_activation` for the output layer.
    """
    layers = []
    last_dim = input_dim
    for layer in net_arch:
        if isinstance(layer, int):
            layers.append(nn.Linear(last_dim, layer))
            layers.append(activation_fn())
            last_dim = layer
        # elif isinstance(layer, str):
        # layers.append(getattr(nn, layer)(**net_kwargs.get(layer, {})))
        elif isinstance(layer, nn.Module):
            layers.append(layer)
        elif isinstance(layer, tuple):
            layers.append(layer[1](**net_kwargs.get(layer[0], {})))
        else:
            raise ValueError("Invalid layer type: {}".format(layer))
    else:
        if output_dim > 0:
            layers.append(nn.Linear(last_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())

    return layers


def _get_schedule(schedule: str,
                  #  total_timesteps: int,
                  initial_value: float,
                  schedule_kwargs: Optional[Dict[str, Any]],
                  resolution: int = 1000) -> Schedule:
    """
    Returns a learning rate schedule.
    :param schedule: (str) The type of scheduler
    :param total_timesteps: (int) The total number of timesteps
    :param initial_value: (float) The initial decaying value
    :param schedule_kwargs: (dict) Extra arguments for the scheduler
    :return: (Schedule) The value scheduler
    """
    dummy_model = nn.Linear(1, 1)
    dummy_optimizer = th.optim.Adam(dummy_model.parameters(), lr=initial_value)
    schedule = getattr(lr_scheduler, schedule)(dummy_optimizer, **schedule_kwargs)
    lrs = []
    for i in range(resolution):
        dummy_optimizer.step()
        schedule.step()
        lrs.append(schedule.get_last_lr()[0])

    # def lr_schedule_fn(progress_remaining: float) -> float:
    #     return lrs[int((1-progress_remaining) * total_timesteps)]
    def lr_schedule_fn(progress_remaining: float) -> float:
        return np.quantile(lrs, progress_remaining)

    return lr_schedule_fn


def get_schedule(schedule:str,
                 initial_value: float,
                 schedule_kwargs: Optional[Dict[str, Any]],
                 warmup: float = 0,
                 resolution: int = 1000) -> Schedule:
    resolution = int(schedule_kwargs.get('resolution', 1000) * (1 - warmup))
    schedule = _get_schedule(schedule=schedule,
                             initial_value=initial_value,
                             schedule_kwargs=schedule_kwargs,
                             resolution=resolution)

    def fn(progress_remaining):
        if 1 - progress_remaining < warmup:
            # return schedule_kwargs['initial_value']
            return initial_value
        else:
            return schedule(progress_remaining + warmup)

    return fn
