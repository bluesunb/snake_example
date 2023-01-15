import numpy as np
import gym

from argparse import ArgumentParser
from gym import spaces


class I024(gym.Env):
    def __init__(self, board_size=4):
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.now = 0
        self.max = 100
        # self.fail_penalty = -1

        self.observation_space = spaces.Box(low=0, high=self.max,
                                            shape=self.board.shape,
                                            dtype=int)

        self.action_space = spaces.Discrete(4)
        self.reset()

    def get_obs(self):
        return self.board.copy()

    def reset(self):
        self.board = self.random_generate(np.zeros_like(self.board, dtype=int), 2)
        self.now = 0
        return self.get_obs()

    def step(self, action):
        info = {}
        self.now += 1
        new_board = self._act(self.board, action)
        done = self.check_fail(new_board, info)

        if not done:
            new_board = self.random_generate(new_board)
            done = self.check_stuck(new_board, info)
            reward = self.calculate_reward(new_board)
            self.board = new_board
            if done and reward == 0:
                reward = self.fail_penalty()
        else:
            reward = self.fail_penalty()

        return self.get_obs(), reward, done, info
    
    def calculate_reward(self, new_board):
        old_board_nonzero = self.board[self.board.nonzero()]
        new_board_nonzero = new_board[new_board.nonzero()]
        
        old_reward = np.max(old_board_nonzero)
        new_reward = np.max(new_board_nonzero)
        
        print(f'old: {old_reward}, new: {new_reward}')
        # return np.log2(np.clip(new_reward - old_reward, 0, np.inf) + 1)
        return np.log2(new_reward ** 2 - old_reward ** 2 + 1)
    
    def fail_penalty(self):
        m = np.max(self.board)
        return - 0.5 * m * (m + 1)
    
    def render(self, mode='human'):
        print('-' * 4 * self.board.shape[1])
        for line in self.board:
            print('|'.join([f'{i if i>0 else "": ^3}' for i in line]), end='|\n')
            print('-' * 4 * self.board.shape[1])
    
    @staticmethod
    def random_generate(board, n=1):
        zero = np.argwhere(board == 0)
        idx = np.random.choice(len(zero), n)
        for i in idx:
            board[tuple(zero[i])] = np.random.choice([1, 2], p=[0.9, 0.1])

        return board

    def check_fail(self, new_board, info):
        done = True
        if np.all(self.board == new_board):
            info['mgs'] = 'Invalid move'
        elif self.now >= self.max:
            info['msg'] = 'Reach max'
        else:
            done = False

        return done

    def check_stuck(self, new_board, info):
        if np.all(new_board):
            h_dup = self._conv(new_board, [1, -1])[:, :-1]
            v_dup = self._conv(new_board.T, [1, -1])[:, :-1]
            if np.all(h_dup) and np.all(v_dup):
                info['msg'] = 'No more move'
                return True
    
    def _push_arr(self, arr):
        l = self.board.shape[0]
        a_sort = np.argsort(arr == 0, axis=1)
        arr = arr[[[i] * l for i in range(l)], a_sort]
        return arr

    def _conv(self, arr, op):
        conv = np.convolve(arr.flatten(), op, 'valid')
        conv = np.concatenate([conv, [0, ]]).reshape(arr.shape)
        return conv

    def _move(self, arr):
        arr = self._push_arr(arr)
        addable = self._conv(arr, [1, -1])[:, :-1]
        addable = (addable == 0).astype(int)

        mask = []
        encounter = np.zeros((addable.shape[0], 1), dtype=int)
        for i in range(addable.shape[1]):
            mask.append(addable[:, i:i+1] ^ encounter)
            encounter |= addable[:, i:i + 1]

        # mask = np.hstack([(1 - addable[:, [0, ]]) if i % 2 else addable[:, [0, ]] for i in range(addable.shape[1])])
        mask = np.hstack(mask)
        addable = np.logical_and(addable, mask)

        r_addable = np.pad(addable, ((0, 0), (0, 1)), 'constant', constant_values=0)
        l_addable = np.pad(addable, ((0, 0), (1, 0)), 'constant', constant_values=0)

        moved = arr.copy()
        moved = np.where(r_addable, (moved + 1) * (moved != 0), moved)
        moved = np.where(l_addable, 0, moved)
        moved = self._push_arr(moved)

        return moved


    def _act(self, arr, op):
        if op == 0:
            return self._move(arr.T).T
        elif op == 1:
            return self._move(arr[:, ::-1])[:, ::-1]
        elif op == 2:
            return self._move(arr[::-1, :].T).T[::-1, :]
        elif op == 3:
            return self._move(arr)


def play():

    action_map = {'w': 0,
                  'd': 1,
                  's': 2,
                  'a': 3}

    env = I024(board_size=4)
    env.reset()
    env.render()
    done, info = False, {}
    while not done:
        action = action_map[input('action: ')]
        obs, reward, done, info = env.step(action)
        env.render()
        print(f'Now: {env.now}, Reward: {reward}')

    print(info)
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    parser.parse_args()
    
    if parser.demo:
        play()
    else:
        pass