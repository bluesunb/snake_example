import gym
import numpy as np

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional, Union
import heuristics

import platform
import os

__version__ = "0.1.0"


class Snake(gym.Env):
    def __init__(self,
                 grid_size=(10, 10),
                 obs_kwargs: Optional[Dict[str, Any]] = None,
                 body_length: Union[int, List[int]] = 3,
                 max_time: int = 0,
                 heuristic: Optional[str] = None,
                 heuristic_kwargs: Optional[Dict[str, Any]] = None,):

        self.__version__ = __version__
        self.body = [(0, 0)]
        self.body_length = body_length
        self.board = np.zeros(grid_size, dtype=np.uint8)

        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.direction = 0
        self.food = (0, 0)
        self.obs_kwargs = obs_kwargs if obs_kwargs is not None else {}

        self.now = 0
        self.last_eat = 0
        self.max_time = max_time if max_time > 0 else 4 * np.sum(self.board.shape)

        if heuristic is None or heuristic == 'distance':
            self.heuristic = heuristics.distance_heuristic
        elif heuristic == 'distance diff':
            self.heuristic = heuristics.distance_diff_heuristic
        elif heuristic == 'angle':
            self.heuristic = heuristics.angle_heuristic
        else:
            raise ValueError("Unknown heuristic: {}".format(heuristic))

        self.heuristic_kwargs = heuristic_kwargs if heuristic_kwargs is not None else {}

        dummy_obs = self.get_obs(**self.obs_kwargs)
        self.observation_space = spaces.Box(low=0, high=1, shape=dummy_obs.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.reset()

    def get_obs(self, **kwargs) -> np.ndarray:
        if kwargs.get("mode", "array") == "array":
            return self._get_obs_array(**kwargs)
        elif kwargs.get("mode", "array") == "coordinate":
            return self._get_obs_coord(**kwargs)
        elif kwargs.get("mode", "array") == "image":
            return self._get_obs_image(**kwargs)

    def _get_obs_array(self, incremental: bool = False) -> np.ndarray:
        obs = np.zeros(self.board.shape, dtype=np.uint8)
        for i, body in enumerate(self.body):
            obs[body] = 1 if not incremental else 1 + i / len(self.body)
        obs = np.concatenate([obs.flatten(), self.food])
        return obs

    def _get_obs_coord(self, normalize: bool = False) -> np.ndarray:
        candidates = self.body[0] + self.direction_vec
        info = np.logical_and(candidates >= 0, candidates < self.board.shape)
        info = np.logical_and(info, self.board[candidates[:, 0], candidates[:, 1]] == 0)
        info = info.astype(np.uint8).tolist()

        obs = list(self.body)
        obs.extend(info)
        obs.extend(self.body)
        obs.extend([(-1, -1)] * (self.board.shape[0] * self.board.shape[1] - len(self.body) - 1))
        obs = np.array(obs) + 1
        if normalize:
            obs = obs / (np.max(self.board.shape) + 1)
        return obs

    def _get_obs_image(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        self.body = self._random_body_generate()
        self.direction = np.random.randint(0, 4)
        self.food = self._random_food_generate()

        self.now = 0
        self.last_eat = 0

        return self.get_obs(**self.obs_kwargs)

    def step(self, action):
        self.now += 1
        self.direction = action

        done, info = False, {}




    def _random_body_generate(self) -> List[Tuple[int, int]]:
        if isinstance(self.body_length, int):
            self.body_length = [self.body_length, self.body_length + 1]
        length = np.random.randint(*self.body_length)
        body = [(np.random.randint(1, self.board.shape[0]-1),
                 np.random.randint(1, self.board.shape[1]-1))]

        board = np.zeros(self.board.shape, dtype=np.uint8)
        board[body[0]] = 1

        def get_candidates(pos):
            candidates = pos + self.direction_vec
            candidates = candidates[(candidates >= 0).all(axis=1) & (candidates < self.board.shape).all(axis=1)]
            candidates = candidates[board[candidates[:, 0], candidates[:, 1]] == 0]
            candidates = [tuple(x) for x in candidates]
            return candidates

        for _ in range(length - 1):
            head_candidates = get_candidates(body[0])
            body_candidates = head_candidates if len(body) == 1 else get_candidates(body[-1])
            if len(head_candidates) == 1 and head_candidates[0] in body_candidates:
                body_candidates.remove(head_candidates[0])
            if len(body_candidates) == 0:
                break
            body.append(body_candidates[np.random.randint(0, len(body_candidates))])
            board[body[-1]] = 1

        return body

    def _random_food_generate(self) -> Tuple[int, int]:
        while True:
            food = (np.random.randint(0, self.board.shape[0]),
                    np.random.randint(0, self.board.shape[1]))
            if food not in self.body:
                return food
