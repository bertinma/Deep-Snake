import model 
from game import Game
import torch
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np

myModel = model.Model().float()

EPOCHS = 1000
GAMMA = 0.9
directions_letters = ['d', 's', 'q', 'z']
direction_str = ['right', 'down', 'left', 'up']
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.0001)
memory = []
results = []


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

def plot_results(results):
    import matplotlib.pyplot as plt
    import numpy as np
    """Plot the results of the training
    Each row is a game, each value in a row is a step and have the same abscissa (step)

    Plot the mean of the results by row in red
    Plot the max of the results by in blue
    Plot each point in black, with small point size.
    """ 
    means = [(i, np.mean(result)) for i, result in enumerate(results)]
    maxs = [(i, np.max(result)) for i, result in enumerate(results)]
    plt.plot([i for i, _ in means], [result for _, result in means], c='red')
    plt.plot([i for i, _ in maxs], [result for _, result in maxs], c='blue')
    for i, result in enumerate(results):
        for r in result:
            plt.scatter(i, r, c='black', s=1)
    plt.show()

last_results = []
best_mean_score = 0 
for epoch in tqdm(range(EPOCHS)):
    if epoch > 0:
        print(game.game_over, steps)
    game = Game(10)
    steps = 0
    tmp = []
    while not game.game_over and steps < 1000:
        old_state = game.get_state()
        # print("Model input : ", old_state)
        old_state_tensor = torch.tensor(old_state, dtype=torch.float32, requires_grad=True)

        with torch.no_grad():
            q_values = myModel(old_state_tensor)
        # game.step(output)
        
        action = torch.argmax(q_values)
        curent_q_value = q_values[action]
        
        direction = get_direction_with_action(action, game.direction)
        print("Game direction : ", game.direction)
        print("Action : ", action)
        print("Direction : ", direction_str[direction])
        next_state = game.compute_next_state(direction)
        reward = game.get_reward()
        if reward > 0:
            print("Score : ", game.score)
            steps = 0
        elif reward < -2:
            print("/!\ Game Over /!\ ")

        if game.is_terminal():
            observed_q_value = reward
        else:
            new_state = torch.tensor(next_state, dtype=torch.float32)
            next_q_value = torch.max(myModel(new_state))
            observed_q_value = reward + (GAMMA * next_q_value)

        myModel.train()
        torch.set_grad_enabled(True)
        target_f = q_values.clone()
        target_f.detach()
        target_f[action] = observed_q_value
        # print("Target : ", target_f)
        # print("Q Values : ", q_values)
        optimizer.zero_grad()
        loss = F.mse_loss(q_values, target_f).requires_grad_(True)
        if reward < 0:
            print("Loss : ", loss)
        # print("Loss : ", loss)
        loss.backward()
        optimizer.step()
        tmp.append(reward)

        # memory 
        steps += 1
        memory.append((old_state, action, reward, next_state, game.is_terminal()))
        game.display_plate()
        if next_state[0] == 1:
            print('Forward danger')
        if next_state[1] == 1:
            print('Right danger')
        if next_state[2] == 1:
            print('Left danger')
        print("\n\n")

    results.append(tmp)
    last_results.append(game.score)

    if len(memory) > 1000:
        memory = random.sample(memory, 100)
    for state, action, reward, next_state, done in memory:
            myModel.train()
            torch.set_grad_enabled(True)
            target = reward
            # print(next_state, state)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True)
            if not done:
                target = reward + GAMMA * torch.max(myModel(next_state_tensor))
            output = myModel(state_tensor)
            target_f = output.clone()
            target_f[action] = target
            target_f.detach()
            optimizer.zero_grad()
            loss = F.mse_loss(output, target_f).requires_grad_(True)
            loss.backward()
            optimizer.step()
    if np.mean(last_results[:-10]) > best_mean_score:
        # save model 
        best_mean_score = np.mean(last_results[:-10])
        torch.save(myModel.state_dict(), "models/model.pt")
        print(results)
        # plot_results(results)
