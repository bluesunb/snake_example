import yaml
import gym
import torch as th

from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from stable_baselines3.common.callbacks import MaybeCallback
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.buffers import ReplayBuffer


def dump_params(params, path):
    with open(path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
        
@dataclass
class LearningParams:
    total_timesteps: int
    callback: MaybeCallback = None
    log_interval: int = 4
    eval_env: Optional[gym.Env] = None
    eval_freq: int = -1
    n_eval_episodes: int = 5
    tb_log_name: str = "run"
    eval_log_path: str = None
    reset_num_timesteps: bool = True
    

@dataclass
class DQNParams:
    learning_rate: Union[float, Schedule] = 1e-4
    buffer_size: int = 1000000,  # 1e
    learning_starts: int = 50000
    batch_size: Optional[int] = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: Union[int, Tuple[int, str]] = 4
    gradient_steps: int = 1
    replay_buffer_class: Optional[ReplayBuffer] = None
    replay_buffer_kwargs: Optional[Dict[str, Any]] = None
    optimize_memory_usage: bool = False
    target_update_interval: int = 10000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10
    tensorboard_log: Optional[str] = None
    create_eval_env: bool = False
    policy_kwargs: Optional[Dict[str, Any]] = None
    verbose: int = 0
    seed: Optional[int] = None
    device: Union[th.device, str] = "auto"