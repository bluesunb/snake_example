from Env.snake_env import Snake
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

env = Snake()
model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
