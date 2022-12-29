import time

import gym
import numpy as np

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional, Union

import platform
import os

__version__ = "0.0.2"

class Snake(gym.Env):
    def __init__(self, grid_size=(12, 12), mode="array", body_length: Union[int, List[int]] = 3):
        self.__version__ = '0.0.2'
        self.body = [(0, 0)]
        self.body_length = body_length
        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.direction = 0
        self.food = (0, 0)
        self.mode = mode
        self.board = np.zeros(grid_size, dtype=np.uint8)
        self.board_size = np.array(grid_size)

        self.now = 0
        self.last_eat = 0
        self.max_time = 4 * self.board_size.sum()

        if self.mode == "array":
            self.observation_space = spaces.Box(low=0, high=self.board_size[0] * self.board_size[1],
                                                shape=(self.board_size[0] * self.board_size[1] + 2, ), dtype=np.uint8)
        elif self.mode == "coord":
            self.observation_space = spaces.Box(low=0, high=self.board_size[0] * self.board_size[1] + 1,
                                                shape=(self.board_size[0] * self.board_size[1] * 2 + 4, ), dtype=np.uint8)
        elif self.mode == "image":
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(3, self.board_size[0], self.board_size[1]), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def get_obs(self):
        if self.mode == "array":
            obs = np.zeros(self.board_size, dtype=np.uint8)
            for i, body in enumerate(self.body, start=1):
                obs[body] = i

            return np.concatenate([obs.flatten(), self.food, ])

        elif self.mode == "coord":
            candidates = self.body[0] + self.direction_vec
            indices = np.arange(4)
            info = np.logical_and((candidates >= 0).all(axis=1), (candidates < self.board.shape).all(axis=1))
            indices = indices[info]
            candidates = candidates[info]
            info = self.board[candidates[:, 0], candidates[:, 1]] == 0
            indices = indices[info]
            info = np.zeros(4, dtype=np.uint8)
            info[indices] = 1

            obs = [self.food]
            obs.extend(self.body)
            obs.extend([(-1, -1)] * (self.board.shape[0] * self.board.shape[1] - len(self.body) - 1))
            obs = np.array(obs) + 1
            obs = np.concatenate([obs.flatten(), info.flatten().astype(np.uint8)])
            # if normalize:
            #     obs = obs / (np.max(self.board.shape) + 1)
            return obs
        elif self.mode == "image":
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            obs[0][self.food] = 255
            for body in self.body:
                obs[1][body] = 255
            obs[2][self.body[0]] = 255
            return obs

    def reset(self):
        self.body = self._random_length_generate()
        self.board = np.zeros(self.board_size, dtype=np.uint8)
        self.board[[t[0] for t in self.body], [t[1] for t in self.body]] = 1

        self.direction = 0
        self.food = self._generate_food()

        self.now = 0
        self.last_eat = 0

        return self.get_obs()

    def step(self, action):
        self.now += 1
        self.direction = action

        if len(self.body) < self.body_length[0]:
            print(1)
            raise ValueError

        done, info = False, {}
        reward = self.heuristic(action)
        # reward = - 1 / self.max_time
        valid, new_head = self._check_head(self.body[0], action, info=info)
        # self.body.pop()
        # self.board[self.body[-1]] -= 1
        self.board[self.body.pop()] -= 1

        if self.now - self.last_eat <= self.max_time and valid:
            self.body.insert(0, new_head)
            self.board[new_head] += 1
            if new_head == self.food:
                self.food = self._generate_food()
                self.last_eat = self.now
                self.body.append(self.body[-1])
                self.board[self.body[-1]] += 1
                reward = 1 * np.sqrt(len(self.body) / 3)
        else:
            done = True
            reward = -1
            if self.now - self.last_eat > self.max_time:
                info["msg"] = "timeout"

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

        if self.mode == "coord":
            render_board = self.board.astype(int)
            food_pos = self.food
        else:
            render_board = self.get_obs()
            # food_pos = render_board[-3:-1]
            food_pos = render_board[-2:]
            # render_board = render_board[:-3].reshape(-1, *self.board_size).squeeze()
            render_board = render_board[:-2].reshape(-1, *self.board_size).squeeze()
        if self.mode == 'image':
            render_board = (render_board[1] - render_board[0]) / 255
        elif self.mode == 'array' or self.mode == 'coord':
            render_board[tuple(food_pos)] = -1
        render_board = np.vectorize(encode)(render_board)
        for row in render_board:
            print(''.join(row))

    # def heuristic(self, direction):
    #     # head = np.array(self.body[0])
    #     # head = np.array(self.body[0])
    #     # food = np.array(self.food)
    #     # return -np.linalg.norm(head - food) / (np.max(self.board_size) * self.max_time)
    #     food_direction = np.array(self.food) - np.array(self.body[0])
    #     if np.all(food_direction == 0):
    #         raise ValueError("Food is on the head")
    #     return np.dot(self.direction_vec[direction], food_direction) / (self.max_time * np.linalg.norm(food_direction))

    def heuristic(self, direction):
        # head = np.array(self.body[0])
        # new_head = head + self.direction_vec[direction]
        # food = np.array(self.food)
        # diff = np.linalg.norm(food - head) - np.linalg.norm(food - new_head)
        # return diff / self.max_time
        food = np.array(self.food)
        head = np.array(self.body[0])
        new_head = head + self.direction_vec[direction]
        return np.linalg.norm(food - new_head) / (self.max_time * self.board_size.max())

    def _random_length_generate(self):
        if isinstance(self.body_length, int):
            self.body_length = [self.body_length, self.body_length + 1]
        length = np.random.randint(*self.body_length)
        body = [(np.random.randint(1, self.board_size[0]-1),
                 np.random.randint(1, self.board_size[1]-1))]

        self.board[body[0]] += 1

        def get_candidates(pos):
            candidates = pos + self.direction_vec
            candidates = candidates[(candidates >= 0).all(axis=1) & (candidates < self.board_size).all(axis=1)]
            candidates = candidates[self.board[candidates[:, 0], candidates[:, 1]] == 0]
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
            self.board[body[-1]] += 1

        return body

    def _check_head(self, head, direction, info=None):
        if not isinstance(head, np.ndarray):
            head = np.array(head)

        vec = self.direction_vec[direction]
        new_head = head + vec
        # valid = np.logical_and(new_head >= 0, new_head < self.board_size).all()
        valid = (new_head >= 0).all() and (new_head < self.board_size).all()
        if not valid:
            if info is not None:
                info['msg'] = "out of bound"
        elif tuple(new_head) in self.body:
            valid = False
            if info is not None:
                info['msg'] = "body"

        return valid, tuple(new_head)

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.board_size[0]),
                    np.random.randint(0, self.board_size[1]))
            if food not in self.body:
                return food


# env = Snake(grid_size=(8, 8), mode="coord")
# obs = env.reset()
# env.render()
# # print(obs)
# print(env.board)
# print(len(env.body))
# done, info = False, {}
# cum_reward = 0
# while not done:
#     action = int(input("action: "))
#     # time.sleep(0.1)
#     # action = np.random.choice(obs[-4:].nonzero()[0])
#     obs, reward, done, info = env.step(action)
#     env.render()
#     # print(obs)
#     print(env.board)
#     print(env.get_obs()[-4:])
#     print(f'reward: {reward:.4f}, cum_reward: {cum_reward:.4f}, len: {len(env.body)}')
#     cum_reward += reward
#
# print(info)