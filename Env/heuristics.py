import numpy as np


def distance_diff_heuristic(head, new_head, food):
    return np.linalg.norm(food - new_head) - np.linalg.norm(food - head)


def distance_heuristic(new_head, food):
    return np.linalg.norm(food - new_head)


def angle_heuristic(head, new_head, food):
    direction = new_head - head
    food_direction = food - head
    return np.dot(direction, food_direction) / (np.linalg.norm(direction) * np.linalg.norm(food_direction))
