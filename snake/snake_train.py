import yaml
import argparse
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

from snake.Env import heuristics
from snake.Env.snake_env import Snake
from param_manager import (
    DQNParams, LearningParams, SnakeParams,
    dump_params, load_params
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--force-now', action='store_true', default=True)
    args = parser.parse_args()

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

    # default config
    env_params = SnakeParams(grid_size=(8, 8),
                             mode='coord',
                             body_length=[4],
                             # heuristic=heuristics.multi_angle_heuristic)
                             heuristic=heuristics.angle_heuristic)

    dqn_params = DQNParams(verbose=1,
                           buffer_size=200000,
                           learning_starts=50000,
                           learning_rate=2e-4,
                           batch_size=256,
                           train_freq=1,
                           exploration_fraction=0.3,
                           exploration_final_eps=0.05,
                           target_update_interval=20000,
                           tensorboard_log='./logs',
                           policy_kwargs=dict(net_arch=[194, 128]))

    learning_params = LearningParams(total_timesteps=200000,
                                     log_interval=10,
                                     eval_freq=500,
                                     n_eval_episodes=10,
                                     tb_log_name=f'snake',
                                     eval_log_path='./logs',
                                     eval_env=Snake(**env_params.__dict__))

    # load config
    if os.path.exists(args.config):
        print('=== Config found! ===')
        params = load_params(args.config)
        env_params.update(**params['env'])
        dqn_params.update(**params['model'])
        learning_params.update(**params['learn'])

    # eval log path setting to save model
    if args.force_now:
        print('=== Force now!: {} ==='.format(time_stamp))
        learning_params.tb_log_name = learning_params.tb_log_name + f'_{time_stamp}'

    # generate log path
    log_path = os.path.join(learning_params.eval_log_path,
                            f'{learning_params.tb_log_name}_1')

    learning_params.eval_log_path = log_path

    # create env
    env = Snake(**env_params.__dict__)

    # create model
    if args.model and os.path.exists(args.model + '.zip'):
        print('=== Model found! ===')
        model = DQN.load(os.path.join(log_path, args.model))
    else:
        model = DQN(MlpPolicy, env, **dqn_params.__dict__)

    model.learn(**learning_params.__dict__)
    model.save(os.path.join(log_path, 'last_model'))

    # save config
    learning_params.eval_env = None
    dump_params(os.path.join(log_path, 'config.yaml'),
                env_params, dqn_params, learning_params)


if __name__ == '__main__':
    # sys.argv = ['train/snake_train.py', ]
    main()
