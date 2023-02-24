import model 
from game import Game
import torch
from tqdm import tqdm
import random
import numpy as np

myModel = model.Model().float()
myModel.load_state_dict(torch.load("models/model.pt"))

GAMMA = 0.9
directions_letters = ['d', 's', 'q', 'z']
direction_str = ['right', 'down', 'left', 'up']


def get_direction_with_action(action, direction):
    # go straight
    if action == 0:
        return direction
    else:
        # initial direction is right 
        if direction == 0:
            # left 
            if action == 1:
                return 3
            # right
            elif action == 2:
                return 1
        # initial direction is down
        elif direction == 1:
            # left 
            if action == 1:
                return 0
            # right
            elif action == 2:
                return 2
        # initial direction is left
        elif direction == 2:
            # left
            if action == 1:
                return 1
            # right
            elif action == 2:
                return 3
        # initial direction is up
        elif direction == 3:
            # left 
            if action == 1:
                return 2
            # right
            elif action == 2:
                return 0


game = Game(10)
while not game.game_over:
    old_state = game.get_state()
    old_state_tensor = torch.tensor(old_state, dtype=torch.float32, requires_grad=True)
    if old_state[0] == 1:
        print("Forward danger")
    if old_state[1] == 1:
        print("Right danger")
    if old_state[2] == 1:
        print("Left danger")
    print("\n\n")

    with torch.no_grad():
        q_values = myModel(old_state_tensor)
    
    action = torch.argmax(q_values)
    curent_q_value = q_values[action]
    
    direction = get_direction_with_action(action, game.direction)
    print("Game direction : ", game.direction)
    print("Action : ", action)
    print("Direction : ", direction_str[direction])
    print("Score : ", game.score)
    next_state = game.compute_next_state(direction)
    reward = game.get_reward()
    game.display_plate()
print("Game over !")