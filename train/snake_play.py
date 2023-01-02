import argparse
import os
import platform
import time

from stable_baselines3 import DQN

from Env.snake_env3 import Snake, __version__
# from Env.snake_env2 import Snake, __version__


home = os.path.expanduser("~")
project_path = os.path.join(home, "PycharmProjects", "snake_example")

def main():
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
        model_path = os.path.join(project_path, '../logs/', args.model)
        # model_path = os.path.join('./logs/train1/', args.model)
    else:
        env_factory = {'grid_size': (8, 8), 'mode': 'array', 'body_length': 5}
        model_path = os.path.join(project_path, '../logs/', args.model)

    env = Snake(**env_factory)
    print(model_path)
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


if __name__ == "__main__":
    main()