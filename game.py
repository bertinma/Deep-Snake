import numpy as np
import random 

DIRECTIONS = ['right', 'down', 'left', 'up']

def to_categorical(labels, num_classes=4):
    """
    Converts integer labels into one-hot encoded categorical labels.

    Arguments:
    labels -- an array of integer labels
    num_classes -- the total number of classes in the dataset

    Returns:
    A one-hot encoded numpy array of shape (len(labels), num_classes)
    """
    assert np.max(labels) < num_classes, "Label value exceeds num_classes"
    
    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    
    return one_hot_labels


class Game():
    def __init__(self, size):
        self.size = size
        self.create_plate()
        self.score = 0
        self.game_over = False
        self.direction = -1

    def new_apple(self):
        while True:
            self.apple = (random.randint(0, self.plate.shape[0]-1), random.randint(0, self.plate.shape[1]-1))
            if self.apple not in self.snake:
                break

    def draw_plate(self):
        self.plate = np.zeros_like(self.plate)
        self.plate[self.apple[0], self.apple[1]] = 2
        for i in self.snake:
            self.plate[i[0], i[1]] = 1

    def create_plate(self):
        self.plate = np.zeros((self.size, self.size))
        
        # init random position of apple and snake
        # apple = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.snake = [(random.randint(0, self.size-1), random.randint(0, self.size-1))]
        while True:
            self.direction = random.randint(0, 3)
            # 0 - right, 1 - down, 2 - left, 3 - up
            if self.direction == 0 and self.snake[0][1] > 0:
                self.snake.append((self.snake[0][0], self.snake[0][1] - 1))
                break
            elif self.direction == 1 and self.snake[0][0] > 0:
                self.snake.append((self.snake[0][0] - 1, self.snake[0][1]))
                break
            elif self.direction == 2 and self.snake[0][1] < self.size - 1:
                self.snake.append((self.snake[0][0], self.snake[0][1] + 1))
                break
            elif self.direction == 3 and self.snake[0][0] < self.size - 1:
                self.snake.append((self.snake[0][0] + 1, self.snake[0][1]))
                break

        self.new_apple()
        self.draw_plate()


    def step(self):
        # 0 - right, 1 - down, 2 - left, 3 - up
        while True:
            new_direction = random.randint(0, 3)
            # if (self.direction == 0 and new_direction != 2) or (self.direction == 1 and new_direction != 3) or (self.direction == 2 and new_direction != 0) or (self.direction == 3 and new_direction != 1):
            #     break
            if not (new_direction + self.direction) % 2 == 0:
                self.direction = new_direction
                break
        
        # move
        if self.direction == 0:
            new_head = (self.snake[0][0] , self.snake[0][1] + 1)
        elif self.direction == 1:
            new_head = (self.snake[0][0] + 1, self.snake[0][1])
        elif self.direction == 2:
            new_head = (self.snake[0][0], self.snake[0][1] - 1)
        elif self.direction == 3:
            new_head = (self.snake[0][0] - 1, self.snake[0][1])


        
        # check if game over
        # if new_head[0] < 0 or new_head[0] >= self.size or new_head[1] < 0 or new_head[1] >= self.size or new_head in self.snake:
        if any(v == 0 or v == self.size for v in new_head) or new_head in self.snake:
            self.game_over = True
            return
        # check if apple eaten
        if new_head == self.apple:
            self.score += 1
            self.new_apple()
            self.snake.append(self.snake[-1])
        
        for i in range(len(self.snake)-1, 0, -1):
            self.snake[i] = self.snake[i-1]
        self.snake[0] = new_head

        self.draw_plate()

    def build_model_input(self):
        input = []
        for block in self.snake:
            input.append(block[0] / self.size)
            input.append(block[1] / self.size)
        for _ in range(47 - len(self.snake)):
            input.append(0)
            input.append(0)
        input.append(self.apple[0] / self.size)
        input.append(self.apple[1] / self.size)
        input.extend(to_categorical([self.direction])[0])
        
        assert len(input) == 100, "Input length is not 100"
        return input

if __name__ == '__main__':
    game = Game(10)
    print(game.plate)

    print(game.apple)
    print(game.snake)

    print(game.score)
    print(game.game_over)
    print(game.direction)

    while not game.game_over:
        game.step()
        print(game.plate)
        print(game.apple)
        print(game.snake)
        print(game.score)
        print(game.game_over)
        print(DIRECTIONS[game.direction])
        input_model = game.build_model_input()
        print(input_model)
        print('\n\n')


