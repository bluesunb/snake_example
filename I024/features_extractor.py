import torch as th
import torch.nn as nn
import gym

from typing import Optional, List, Tuple, Type
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 64,
                 net_arch: List[int] = None,
                 kernel_arch: List[int] = None,
                 stride_arch: List[int] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU):
        super(CNNExtractor, self).__init__(observation_space, features_dim)

        if net_arch is None:
            net_arch = [32, 64, 64]
        if kernel_arch is None:
            kernel_arch = [3] * len(net_arch)
        if stride_arch is None:
            stride_arch = [1] * len(net_arch)

        layers = []
        n_input_channels = observation_space.shape[0]
        for i, (n_filters, kernel_size, stride) in enumerate(zip(net_arch, kernel_arch, stride_arch)):
            layer = nn.Conv2d(n_input_channels, n_filters, kernel_size=kernel_size, stride=stride)
            layers.append(layer)
            layers.append(activation_fn())
            n_input_channels = n_filters

        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CNNExtractor2(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 64,
                 kernels: List[int] = (2, 3, 4),):
        super().__init__(observation_space, features_dim)
        n_input_channels = 1

        self.kernel_arch = kernels
        for i, kernel_size in enumerate(kernels):
            layer = nn.Conv2d(n_input_channels, n_input_channels, kernel_size=kernel_size, stride=1, padding='same')
            setattr(self, f'preprocess{i}', layer)

        self.cnn = nn.Sequential(
            nn.Conv2d(len(kernels) * n_input_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, features_dim, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.randn(1, len(kernels) * n_input_channels, *observation_space.shape[-2:])
            ).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = th.cat([getattr(self, f'preprocess{i}')(observations) for i in range(len(self.kernel_arch))], dim=1)
        return self.linear(self.cnn(x))