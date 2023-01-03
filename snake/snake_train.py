import yaml
import argparse
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

from snake.Env.snake_env import Snake
from param_manager import DQNParams, LearningParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--force-now', action='store_true', default=True)
    args = parser.parse_args()

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

    # default config
    env_config = dict(grid_size=(8, 8),
                      mode='coord',
                      body_length=[5],
                      heuristic='multi_angle_heuristic')

    dqn_config = dict(verbose=1,
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
    
    learning_config = dict(total_timesteps=200000,
                           log_interval=10,
                           eval_freq=500,
                           n_eval_episodes=10,
                           tb_log_name=f'snake',
                           eval_log_path='./logs')

    # load config
    if os.path.exists(args.config):
        print('=== Config found! ===')
        params = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        dqn_config.update(params.get('dqn', {}))
        learning_config.update(params.get('learning', {}))

    # eval log path setting to save model
    if args.force_now:
        print('=== Force now!: {} ==='.format(time_stamp))
        learning_config['tb_log_name'] = learning_config['tb_log_name'] + f'_{time_stamp}'

    log_path = os.path.join(learning_config['eval_log_path'],
                            f'{learning_config["tb_log_name"]}_1')

    learning_config['eval_log_path'] = log_path


    # create env, parameters
    dqn_params = DQNParams(**dqn_config)
    learning_params = LearningParams(**learning_config)
    env = Snake(**env_config)


    # create model
    if args.model and os.path.exists(args.model + '.zip'):
        print('=== Model found! ===')
        model = DQN.load(os.path.join(log_path, args.model))
    else:
        model = DQN(MlpPolicy, env, **dqn_params.__dict__)

    model.learn(eval_env=Snake(**env_config),
                **learning_params.__dict__)
    model.save(os.path.join(log_path, 'last_model'))

    # save config
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        yaml.dump({'env': env_config,
                   'dqn': dqn_params.__dict__,
                   'learning': learning_params.__dict__}, f)


if __name__ == '__main__':
    # sys.argv = ['train/snake_train.py', ]
    main()