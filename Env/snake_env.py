import gym
import numpy as np

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional

import platform
import os

__version__ = "0.0.1"

class Snake(gym.Env):
    def __init__(self, grid_size=(12, 12), mode="array"):
        self.body = None
        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.direction = None
        self.food = None
        self.mode = mode
        self.board_size = np.array(grid_size)

        self.now = 0
        self.last_eat = 0
        self.max_time = 4 * self.board_size.sum()

        if self.mode == 'array':
            self.observation_space = spaces.Box(low=0, high=np.max(self.board_size),
                                                shape=(grid_size[0] * grid_size[1] + 3,), dtype=int)
        elif self.mode == 'image':
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(3, self.board_size[0], self.board_size[1]), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def get_obs(self):
        def _get_linear_pos(coord):
            return coord[0] * self.board_size[0] + coord[1]

        if self.mode == 'array':
            obs = np.zeros(self.board_size, dtype=int)
            for body in self.body:
                obs[body] = 1
            return np.concatenate(
                [obs.flatten(), self.food, [_get_linear_pos(self.body[0]) > _get_linear_pos(self.body[-1])]])

        elif self.mode == 'image':
            obs = np.zeros(self.observation_space.shape, dtype=int)
            obs[0][self.food] = 255
            for body in self.body:
                obs[1][body] = 255
            obs[2][self.body[0]] = 255
            return obs

    def reset(self):
        self.body = [(np.random.randint(1, self.board_size[0] - 1),
                      np.random.randint(1, self.board_size[1] - 1))]

        vecs = []
        for _ in range(2):
            while True:
                vec = self.direction_vec[np.random.choice(4)]
                pos = np.array(self.body[-1]) + vec
                if np.logical_and(pos >= 0, pos < self.board_size).all() and tuple(pos) not in self.body:
                    vecs.append(vec)
                    self.body.append(tuple(pos))
                    break

        self.direction = np.random.choice(
            [i for i, vec in enumerate(self.direction_vec) if np.all(vec != -vecs[0])]
        )
        self.food = self._generate_food()

        self.now = 0
        self.last_eat = 0

        return self.get_obs()

    def next_pos(self, action: Optional[int] = None) -> Tuple[np.ndarray, bool]:
        head = np.array(self.body[0])
        if action is None:
            action = self.direction

        if action == 0:
            new_head = head - np.array([1, 0])
        elif action == 1:
            new_head = head + np.array([0, 1])
        elif action == 2:
            new_head = head + np.array([1, 0])
        elif action == 3:
            new_head = head - np.array([0, 1])
        else:
            raise ValueError("Invalid action")

        self.body.pop()

        grown = False
        if tuple(new_head) == self.food:
            self.last_eat = self.now
            grown = True

        return new_head, grown

    def step(self, action: int):
        self.now += 1
        self.direction = action

        done, info = False, {}
        # reward = self.heuristic(action) - 0.01
        reward = -1 / self.max_time + self.heuristic(action)
        new_head, grown = self.next_pos(action)

        if not np.logical_and(new_head >= 0, new_head < self.board_size).all():
            reward = -1
            done = True
            info['msg'] = 'out of bounds'
        elif tuple(new_head) in self.body[1:]:
            reward = -1
            done = True
            info['msg'] = 'body'
        elif self.now - self.last_eat > self.max_time:
            reward = -1
            done = True
            info['msg'] = 'timeout'
        else:
            self.body.insert(0, tuple(new_head))
            if grown:
                self.food = self._generate_food()
                self.body.append(self.body[-1])
                reward = 1

        return self.get_obs(), reward, done, info

    def render(self, mode="human"):
        if mode == "char":
            black_square = chr(9608) * 2
            white_square = chr(9617) * 2
            # food = chr(9679) * 2
            food = chr(9675) * 2
        else:
            black_square = chr(int('2b1b', 16))
            white_square = chr(int('2b1c', 16))
            food = chr(int('1f34e', 16))
            # food = chr(int('1f7e7', 16))

        def encode(v):
            if v == 0:
                return white_square
            elif v > 0:
                return black_square
            elif v == -1:
                return food

        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

        render_board = self.get_obs().copy()
        food_pos = render_board[-3:-1]
        render_board = render_board[:-3].reshape(-1, *self.board_size).squeeze()
        if self.mode == 'image':
            render_board = (render_board[1] - render_board[0]) / 255
        elif self.mode == 'array':
            render_board[tuple(food_pos)] = -1
        render_board = np.vectorize(encode)(render_board)
        for row in render_board:
            print(''.join(row))

    def heuristic(self, direction):
        head = np.array(self.body[0])
        new_head = head + self.direction_vec[direction]
        food = np.array(self.food)
        diff = np.linalg.norm(food - head) - np.linalg.norm(food - new_head)
        return diff / self.max_time
        # food_direction = np.array(self.food) - np.array(self.body[0])
        # if np.all(food_direction == 0):
        #     raise ValueError("Food is on the head")
        # return np.dot(self.direction_vec[direction], food_direction) / (self.max_time * np.linalg.norm(food_direction))

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.board_size[0]),
                    np.random.randint(0, self.board_size[1]))
            if food not in self.body:
                return food


# gym.envs.registry.register(
#     id='Snake-v0',
#     entry_point=Snake,
#     max_episode_steps=1000,
# )

# env = Snake(grid_size=(9, 9))
# obs = env.reset()
# env.render()
# print(obs)
# done, info = False, {}
# cum_reward = 0
# while not done:
#     action = int(input('action: '))
#     obs, reward, done, info = env.step(action)
#     env.render()
#     print(obs)
#     print(f'reward: {reward}')
#     cum_reward += reward

# print(cum_reward, info)
