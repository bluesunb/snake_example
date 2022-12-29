from datetime import datetime
from DQN.dqn import DQN
from DQN.features_extractor import SnakeFeaturesExtractor
from DQN.policy import DQNPolicy
from DQN.utils import _get_schedule
from stable_baselines3.common.utils import get_linear_fn


from Env.snake_env import Snake

# env = Snake(grid_size=(8, 8), mode='array', reset_kwargs={'body_length': 5})
env = Snake(grid_size=(8, 8), mode='array')
eval_env = Snake(grid_size=(8, 8), mode='array')

feature_extractor_kwargs = {"features_dim": 128}
policy_kwargs = {'net_arch': [64,],
                 'normalize_images': False,
                 'features_extractor_class': SnakeFeaturesExtractor,
                 'features_extractor_kwargs': feature_extractor_kwargs}

eps_schedule = get_linear_fn(1.0, 0.05, 0.3)

lr_schedule = _get_schedule('StepLR',
                            initial_value=1e-3,
                            schedule_kwargs={'step_size': 300, 'gamma': 0.1},
                            resolution=1000)

model_kwargs = dict(policy=DQNPolicy,
                    env=env,
                    learning_rate=lr_schedule,
                    buffer_size=int(1e5),
                    learning_starts=5e4,
                    batch_size=256,
                    tau=0.99,
                    gamma=0.99,
                    train_freq=(1, 'step'),
                    gradient_steps=1,
                    target_update_interval=1e4,
                    exploration_rate=eps_schedule,
                    tensorboard_log="./logs/train1_tensorboard/",
                    create_eval_env=True,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    seed=1004)

learn_kwargs = dict(total_timesteps=2e5,
                    log_interval=10,
                    eval_env=env,
                    eval_freq=1000,
                    n_eval_episodes=10,
                    tb_log_name=f"snake_dqn_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    eval_log_path="./logs/train1/")

model = DQN(**model_kwargs)

print("model structure:")
print(model.policy)
input("Press Enter to continue...")

model.learn(**learn_kwargs)