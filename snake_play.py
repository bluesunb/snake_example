import argparse
import os
import platform
import time

from stable_baselines3 import DQN

from Env.snake_env import Snake, __version__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='human')
    # parser.add_argument('--seed', type=int, default=1004)
    args = parser.parse_known_args()[0]

    platform_name = platform.system()
    print(platform_name)
    print(args)

    if __version__ == '0.0.1':
        env_factory = {'grid_size': (8, 8), 'mode': 'array'}
    else:
        env_factory = {'grid_size': (8, 8), 'mode': 'array', 'body_length': 5}

    env = Snake(**env_factory)
    model = DQN.load(args.model)

    obs = env.reset()
    cum_reward = 0
    heuristic_reward = [0, 0]
    done, info = False, {}
    i = 0

    env.render(mode=args.mode)

    while not done and i < args.max_steps:
        time.sleep(0.2)
        if platform_name == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_reward += reward
        if not done and reward < 1:
            if reward < 0:
                heuristic_reward[0] += reward
            else:
                heuristic_reward[1] += reward
        env.render(mode=args.mode)
        i += 1
        print(f'reward: {reward}, length: {len(env.body)}')
    print(f"cumulative reward: {cum_reward:.4f}, heuristic reward: {[round(x, 4) for x in heuristic_reward]}")
    print(f"info: {info}")


if __name__ == "__main__":
    main()
