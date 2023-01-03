from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

# from Env.snake_env import Snake, __version__
from snake.Env import Snake, __version__

env_factory = {}
if __version__ == '0.0.1':
    env_factory = {'grid_size': (8, 8), 'mode': 'array'}
# elif __version__ == '0.0.2':
else:
    env_factory = {'grid_size': (8, 8), 'mode': 'coord', 'body_length': [5]}

env = Snake(**env_factory)
eval_env = Snake(**env_factory)

learn_kwargs = dict(total_timesteps=200000,
                    log_interval=10,
                    eval_env=eval_env,
                    eval_freq=500,
                    n_eval_episodes=10,
                    tb_log_name=f'snake_dqn_{datetime.now().strftime("%Y%m%d_%H%M")}',
                    eval_log_path='./logs/train2/')

model = DQN(MlpPolicy, env,
            verbose=1,
            buffer_size=200000,
            learning_starts=50000,
            learning_rate=1e-4,
            # learning_rate=get_schedule(schedule='MultiplicativeLR',
            #                            initial_value=1e-3,
            #                            schedule_kwargs={'lr_lambda': lambda e: 0.99},
            #                            warmup=0.4,
            #                            resolution=1000),
            tensorboard_log="./logs/train2_tensorboard/",
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            batch_size=256,
            train_freq=1,
            target_update_interval=20000,
            policy_kwargs={'net_arch': [128, 128]})

print("model_structure")
print(model.policy)
# input("Press Enter to continue...")

# model.learn(progress_bar=True, **learn_kwargs)
model.learn(**learn_kwargs)
