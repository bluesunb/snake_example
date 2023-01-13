import argparse
import os, sys
import platform
import time
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from Env.snake_env import Snake
from param_manager import load_params

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

    env = Snake(**env_params)
    model = DQN.load(model_path)

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
        obs, reward, done, info = env.step(int(action))
        cum_reward += reward
        if not done and reward < 1:
            if reward < 0:
                heuristic_reward[0] += reward
            else:
                heuristic_reward[1] += reward
        env.render(mode=args.mode)
        i += 1
        print(f'reward: {reward:.5e}, cum_reward: {cum_reward:.5e}',
              f' heuristic_reward: {[round(x, 4) for x in heuristic_reward]}')
    # print(f"cumulative reward: {cum_reward:.4f}, heuristic reward: {[round(x, 4) for x in heuristic_reward]}")
    print(f"info: {info}")

    if args.eval:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print(f'Evaluation:')
        print(f'====================')
        print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")


if __name__ == "__main__":
    # sys.argv.append('logs/snake_20230113_1559_1/')
    main()
