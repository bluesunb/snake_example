import numpy as np
from snake_env3 import Snake


def distance_diff_heuristic(env: Snake):
    head = env.body[0]
    new_head = head + env.direction_vec[env.direction]
    food = env.food
    return np.linalg.norm(food - new_head) - np.linalg.norm(food - head)


def distance_heuristic(env: Snake):
    new_head = env.body[0] + env.direction_vec[env.direction]
    food = env.food
    return np.linalg.norm(food - new_head)


def angle_heuristic(env: Snake):
    head= env.body[0]
    new_head = head + env.direction_vec[env.direction]
    food = env.food
    direction = new_head - head
    food_direction = food - head
    return np.dot(direction, food_direction) / (np.linalg.norm(direction) * np.linalg.norm(food_direction))


def multi_angle_heuristic(self, body, food, direction):
    head = np.array(body[0])
    tail = np.array(body[-1])
    food = np.array(food)

    vec = self.direction_vec[direction]
    vec_food = food - head
    vec_tail = tail - head
    vec_tail = vec_tail / np.linalg.norm(vec_tail)
    vec_food = vec_food / np.linalg.norm(vec_food)
    vec = vec / np.linalg.norm(vec)
    return np.dot(vec, vec_food) + 0.5 * np.dot(vec, vec_tail)