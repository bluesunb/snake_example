import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from snake_env3 import Snake, __version__

model1 = DQN.load("../../logs/train2/best_model_1.zip")
model2 = DQN.load("../../logs/train2/best_model.zip")

eval_env = Snake(grid_size=(8, 8), mode='coord', body_length=[5, 10, 15])

mean_reward1, std_reward1 = evaluate_policy(model1, eval_env, n_eval_episodes=100)
mean_reward2, std_reward2 = evaluate_policy(model2, eval_env, n_eval_episodes=100)

print(f"mean_reward1: {mean_reward1}, std_reward1: {std_reward1}")
print(f"mean_reward2: {mean_reward2}, std_reward2: {std_reward2}")