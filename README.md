# Snake RL with Pytorch 

This is a simple implementation of a reinforcement learning agent that learns to play the game Snake. The agent is trained using the [Pytorch](https://pytorch.org/) library. The game is implemented using console prints.

## 1. Model architecture and RL Training 
- Q_Learnign 
- MSE loss
- Adam optimizer
- Model architecture:
    - 2 hidden layers with 50 neurons each
    - 1 output layer with 3 neurons for straight, left and right


## 2. Build and run the docker image

```console 
docker build -t snake_rl .
docker run -it snake_rl -v models:/app/models
```

## 3. Train the model

```console
python train.py
```

## 4. Play the game

```console
python play.py
```
