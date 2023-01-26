import argparse
import os, sys
import platform
import time
import yaml
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from I024_env import I024
from param_manager import load_params
import features_extractor

home = os.path.expanduser("~")
project_path = os.path.join(home, "PycharmProjects", "snake_example")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='human')
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--seed', type=int, default=1004)
    args, paths = parser.parse_known_args()

    print(args)
    print(paths)

    platform_name = platform.system()
    print(platform_name)
    print(args)

    if len(paths) == 1:
        model_path = paths[0] + '/best_model.zip'
        config_path = paths[0] + '/config.yaml'
    else:
        model_path, config_path = paths
    config = load_params(config_path)
    # config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    env_params = config['env']

    env = I024(**env_params)
    model = DQN.load(model_path)

    obs = env.reset()
    cum_reward = 0
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
        obs, reward, done, info = env.step(int(action))
        cum_reward += reward
        env.render(mode=args.mode)
        i += 1
        print(f'reward: {reward:.5e}, cum_reward: {cum_reward:.5e}')
    # print(f"cumulative reward: {cum_reward:.4f}, heuristic reward: {[round(x, 4) for x in heuristic_reward]}")
    print(f"info: {info}")

    if args.eval:
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        ep_rewards = []
        ep_max_legnth = []

        for i in tqdm(range(500)):
            obs = env.reset()
            done, info = False, {}
            cum_reward = 0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(int(action))
                cum_reward += reward
            ep_rewards.append(cum_reward)
            ep_max_legnth.append(env.board.max())

        print(f'Evaluation:')
        print(f'====================')
        print(f"mean_reward: {np.mean(ep_rewards)}, "
              f"std_reward: {np.std(ep_rewards)}, "
              f"mean_length: {np.mean(ep_max_legnth)}")

        with open(os.path.join(os.path.dirname(model_path), 'eval.txt'), 'w') as f:
            f.write(f'mean reward: {np.mean(ep_rewards)}\n'
                    f'std reward: {np.std(ep_rewards)}\n'
                    f'mean length: {np.mean(ep_max_legnth)}')


if __name__ == "__main__":
    # sys.argv.extend(["logs/1024_20230115_1515_1/last_model.zip", "logs/1024_20230115_1515_1/config.yaml"])
    main()
