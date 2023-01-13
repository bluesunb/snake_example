import gym
import numpy as np
import argparse

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import platform
import os

__version__ = "0.0.3"


class Snake(gym.Env):
    def __init__(self,
                 grid_size=(12, 12),
                 mode="coord",
                 body_length: Union[int, List[int]] = 3,
                 heuristic: Callable = None,
                 heuristic_kwargs: Optional[Dict[str, Any]] = None):

        self.__version__ = '0.0.3'
        self.body = [(0, 0)]
        self.body_length = body_length
        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.direction = 0
        self.food = (0, 0)
        self.before_food = self.food
        self.mode = mode
        self.board = np.zeros(grid_size, dtype=np.uint8)
        self.board_size = np.array(grid_size)

        self.heuristic = heuristic if heuristic is not None else lambda x: 0
        self.heuristic_kwargs = {} if heuristic_kwargs is None else heuristic_kwargs

        self.now = 0
        self.last_eat = 0
        self._max_time = 4 * self.board_size.sum()

        self.observation_space = spaces.Box(low=0, high=self.board_size[0] * self.board_size[1] + 1,
                                            shape=(self.board_size[0] * self.board_size[1] * 2 + 4, ), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.reset()

    @property
    def max_time(self):
        return self._max_time + 2 * len(self.body)

    def get_obs(self):
        """
        obs: [food_coord, snake_body_coord, padding, actions_can_take(one-hot)]
        """
        candidates = self.body[0] + self.direction_vec
        indices = np.arange(4)
        bound_condition = np.logical_and((candidates >= 0).all(axis=1), (candidates < self.board.shape).all(axis=1))
        indices = indices[bound_condition]
        candidates = candidates[bound_condition]
        body_condition = self.board[candidates[:, 0], candidates[:, 1]] == 0
        indices = indices[body_condition]
        action_mask = np.zeros(4, dtype=np.uint8)
        action_mask[indices] = 1

        obs = [self.food]
        obs.extend(self.body)
        obs.extend([(-1, -1)] * (self.board.shape[0] * self.board.shape[1] - len(self.body) - 1))
        obs = np.array(obs) + 1
        obs = np.concatenate([obs.flatten(), action_mask])
        return obs

    def reset(self):
        self.board = np.zeros(self.board_size, dtype=np.uint8)
        self.body = self._random_length_generate()
        self.board[[t[0] for t in self.body], [t[1] for t in self.body]] = 1
        self.direction = 0
        self.food = self._generate_food()

        self.now = 0
        self.last_eat = 0

        return self.get_obs()

    def render(self, mode="human"):
        if mode == "char":
            black_square = chr(9608) * 2
            white_square = chr(9617) * 2
            # food = chr(9679) * 2
            food = chr(9675) * 2
        else:
            black_square = chr(int('2b1b', 16))
            white_square = chr(int('2b1c', 16))
            before_food = chr(int('1f7e9', 16))
            food = chr(int('1f34e', 16))
            # food = chr(int('1f7e7', 16))

        def encode(v):
            if v == 0:
                return white_square
            elif v > 0:
                return black_square
            elif v == -1:
                return food
            elif v == -2:
                return before_food

        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

        render_board = self.board.astype(int).copy()
        food_pos = self.food
        render_board[food_pos] = -1
        render_board[self.before_food] = -2
        render_board = np.vectorize(encode)(render_board)
        for row in render_board:
            print(''.join(row))

    def step(self, action):
        self.now += 1
        self.direction = action

        done, info = False, {}
        reward = 0
        new_head = self.body[0] + self.direction_vec[self.direction]
        self.board[self.body.pop()] -= 1

        bound_condition = (new_head >= 0).all() and (new_head < self.board_size).all()
        body_condition = tuple(new_head) not in self.body
        starve_condition = self.now - self.last_eat <= self.max_time
        if bound_condition and body_condition and starve_condition:
            self.body.insert(0, tuple(new_head))
            self.board[tuple(new_head)] = 1
            if self.food == tuple(new_head):
                reward = 10
                self.last_eat = self.now
                self.body.append(self.body[-1])
                self.board[self.body[-1]] += 1
                self.food = self._generate_food()
            else:
                reward = self.heuristic(self, **self.heuristic_kwargs)
        else:
            done = True
            reward = -100
            if not bound_condition:
                msg = 'out of board'
            elif not body_condition:
                msg = 'body'
            else:
                msg = 'timeout'
            info['mgs'] = msg
        return self.get_obs(), reward, done, info

    def _random_length_generate(self):
        if isinstance(self.body_length, int):
            self.body_length = [self.body_length]
        length = np.random.choice(self.body_length)
        body = [(np.random.randint(1, self.board_size[0]-1),
                 np.random.randint(1, self.board_size[1]-1))]

        self.board[body[0]] = 1

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
            self.board[body[-1]] = 1

        return body

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.board_size[0]),
                    np.random.randint(0, self.board_size[1]))
            if food not in self.body:
                self.before_food = self.food
                return food
            
    def __dict__(self):
        params = {'mode': self.mode,
                  'body_length': self.body_length,
                  'heuristic': self.heuristic,
                  'heuristic_kwargs': self.heuristic_kwargs}
        
        return params

def play_env():
    env = Snake(grid_size=(8, 8), mode="coord")
    obs = env.reset()
    env.render()
    # print(obs)
    print(env.board)
    print(len(env.body))
    done, info = False, {}
    cum_reward = 0
    while not done:
        action = int(input("action: "))
        # time.sleep(0.1)
        # action = np.random.choice(obs[-4:].nonzero()[0])
        obs, reward, done, info = env.step(action)
        env.render()
        # print(obs)
        print(env.board)
        print(env.get_obs()[-4:])
        print(f'reward: {reward:.4f}, cum_reward: {cum_reward:.4f}, len: {len(env.body)}')
        cum_reward += reward

    print(info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()
    if args.demo:
        play_env()
    else:
        pass


if __name__ == '__main__':
    main()