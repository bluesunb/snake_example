import gym
import torch as th
from torch import nn

from typing import Union, List, Optional, Type, Dict, Any, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SnakeFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor for the Snake environment.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(SnakeFeaturesExtractor, self).__init__(observation_space, features_dim)

        # Create the network
        if observation_space.shape[0] == 3:
            input_dim = observation_space.shape[0]
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            # input_dim = observation_space.shape[0] * observation_space.shape[1]
            input_dim = observation_space.shape[0]
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(input_dim),
                nn.Linear(in_features=input_dim, out_features=256),
                nn.ReLU(),
            )

        with th.no_grad():
            n_flatten = self.net(th.as_tensor(observation_space.sample()[None]).float()).shape[-1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.net(observations))
