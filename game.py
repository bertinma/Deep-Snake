import numpy as np
import random 

DIRECTIONS = ['right', 'down', 'left', 'up']
DIRECTIONS_LETTERS = ['d', 's', 'q', 'z']


def compute_distance(apple, snake_part):
    """
    Computes the distance between the apple and the head of the snake.

    Arguments:
    apple -- a tuple of the form (x, y) representing the position of the apple
    snake -- a list of tuples of the form (x, y) representing the position of the snake

    Returns:
    A float representing the distance between the apple and the head of the snake
    """
    return np.sqrt((apple[0] - snake_part[0])**2 + (apple[1] - snake_part[1])**2)


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
        self.score = 0
        self.game_over = False
        self.direction = -1
        self.create_plate()

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

    def display_plate(self):
        # replace 0 by ' ' and 1 by 'X' and 2 by 'O'
        plate = np.where(self.plate == 0, '.', self.plate)
        plate = np.where(plate == '1.0', 'X', plate)
        plate = np.where(plate == '2.0', 'O', plate)
        if not any(v == -1 or v == self.size for v in self.snake[0]):
            plate[self.snake[0][0], self.snake[0][1]] = 'x'
        for i in range(plate.shape[0]):
            print(' '.join(plate[i]))
            # print('\n')


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
            # new_direction = random.randint(0, 3)
            in_str = input("Enter direction: ")
            if in_str not in DIRECTIONS_LETTERS:
                print("Wrong direction")
                continue
            new_direction = DIRECTIONS_LETTERS.index(in_str)
            # if (self.direction == 0 and new_direction != 2) or (self.direction == 1 and new_direction != 3) or (self.direction == 2 and new_direction != 0) or (self.direction == 3 and new_direction != 1):
            #     break
            if self.direction == new_direction:
                break
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
        if any(v == -1 or v == self.size for v in new_head) or new_head in self.snake:
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

    # def build_model_input(self):
    #     input = []
    #     for block in self.snake:
    #         input.append(block[0] / self.size)
    #         input.append(block[1] / self.size)
    #     for _ in range(47 - len(self.snake)):
    #         input.append(0)
    #         input.append(0)
    #     input.append(self.apple[0] / self.size)
    #     input.append(self.apple[1] / self.size)
    #     input.extend(to_categorical([self.direction])[0])
        
    #     assert len(input) == 100, "Input length is not 100"
    #     return input

    def get_state(self):
        """
        Return the state.
        The state is a numpy array of 11 values, representing:
            - Danger 1 OR 2 steps ahead
            - Danger 1 OR 2 steps on the right
            - Danger 1 OR 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side      
        """

        state = [
            # Danger straight
            (self.direction == 0 and ((self.snake[0][1] == self.size - 1) or any([self.snake[0] == (s[0], s[1] - 1) for s in self.snake[3:]]))) or
            (self.direction == 1 and ((self.snake[0][0] == self.size - 1) or any([self.snake[0] == (s[0] - 1, s[1]) for s in self.snake[3:]]))) or
            (self.direction == 1 and ((self.snake[0][1] == 0) or any([self.snake[0] == (s[0], s[1] + 1) for s in self.snake[3:]]))) or  
            (self.direction == 3 and ((self.snake[0][0] == 0) or any([self.snake[0] == (s[0] + 1, s[1]) for s in self.snake[3:]]))),

            # Danger right
            (self.direction == 0 and ((self.snake[0][0] == self.size - 1) or any([self.snake[0] == (s[0] - 1, s[1]) for s in self.snake[3:]]))) or
            (self.direction == 1 and ((self.snake[0][1] == 0) or any([self.snake[0] == (s[0], s[1] + 1) for s in self.snake[3:]]))) or
            (self.direction == 1 and ((self.snake[0][0] == 0) or any([self.snake[0] == (s[0] + 1, s[1]) for s in self.snake[3:]]))) or
            (self.direction == 3 and ((self.snake[0][1] == self.size - 1) or any([self.snake[0] == (s[0], s[1] - 1) for s in self.snake[3:]]))),
            # Danger left
            (self.direction == 0 and ((self.snake[0][0] == 0) or any([self.snake[0] == (s[0] + 1, s[1]) for s in self.snake[3:]]))) or
            (self.direction == 1 and ((self.snake[0][1] == self.size - 1) or any([self.snake[0] == (s[0], s[1] - 1) for s in self.snake[3:]]))) or
            (self.direction == 1 and ((self.snake[0][0] == self.size - 1) or any([self.snake[0] == (s[0] - 1, s[1]) for s in self.snake[3:]]))) or
            (self.direction == 3 and ((self.snake[0][1] == 0) or any([self.snake[0] == (s[0], s[1] + 1) for s in self.snake[3:]]))),

            # Apple forward
            (self.direction == 0 and self.apple == (self.snake[0][0], self.snake[0][1] + 1)) or
            (self.direction == 1 and self.apple == (self.snake[0][0] + 1, self.snake[0][1])) or
            (self.direction == 2 and self.apple == (self.snake[0][0], self.snake[0][1] - 1)) or
            (self.direction == 3 and self.apple == (self.snake[0][0] - 1, self.snake[0][1])),

            self.direction == 0,
            self.direction == 1,
            self.direction == 2,
            self.direction == 3,
            # Food location
            self.apple[0] < self.snake[0][0],  # food up   
            self.apple[0] > self.snake[0][0],  # food down
            self.apple[1] < self.snake[0][1],  # food left
            self.apple[1] > self.snake[0][1]  # food right
        ]
        return np.array(state, dtype=int)

    def get_reward(self):
        if self.game_over:
            return -10
        elif self.snake[0] == self.apple:
            return 10
        else:
            if compute_distance(self.snake[0], self.apple) < compute_distance(self.snake[1], self.apple):
                return 2
            else:
                return -2

    def compute_next_state(self, new_direction):
        # if not (new_direction + self.direction) % 2 == 0:
        self.direction = new_direction
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
        if any(v == -1 or v == self.size for v in new_head) or new_head in self.snake:
            self.game_over = True
        # check if apple eaten
        if new_head == self.apple:
            self.score += 1
            self.new_apple()
            self.snake.append(self.snake[-1])
        
        for i in range(len(self.snake)-1, 0, -1):
            self.snake[i] = self.snake[i-1]
        self.snake[0] = new_head
        if not self.game_over:
            self.draw_plate()
        return self.get_state()

    def is_terminal(self):
        return self.game_over


if __name__ == '__main__':
    game = Game(10)

    while not game.game_over:
        game.display_plate()
        print("Apple position : ", game.apple)
        print("Snake position : ", game.snake)
        print("Score : ", game.score)
        print("Game over : ", game.game_over)
        print("Direction : ", DIRECTIONS[game.direction])
        input_model = game.build_model_input()
        print("Model input : ", input_model)
        print('\n\n')
        game.step()


