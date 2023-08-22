import numpy as np
import random
from keras import layers, optimizers, Sequential
import matplotlib.pyplot as plt


def draw_chart(iteration, reward, name):
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.scatter(iteration, reward, c='r', marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    plt.show()
    ax.figure.savefig(f'{name}.png')


def random_with_possibility(epsilon):
    r = random.random()
    return r < epsilon


def discretize_state(state):
    discrete_state = (min(2, max(-2, int((state[0]) / 0.05))), min(2, max(-2, int((state[1]) / 0.1))),
                      min(2, max(-2, int((state[2]) / 0.1))), min(2, max(-2, int((state[3]) / 0.1))),
                      min(2, max(-2, int((state[4]) / 0.1))), min(2, max(-2, int((state[5]) / 0.1))), int(state[6]),
                      int(state[7]))
    return discrete_state


def save_table(qtable):
    with open('Qtable.txt', 'w') as f:
        f.write(str(qtable))


def load_table():
    try:
        with open('Qtable.txt', 'r') as f:
            return eval(f.read())
    except FileNotFoundError:
        print("Qtable file not found. Creating a new Qtable.")
        return Qtable()


def save_weights(weights):
    with open('Weights.txt', 'w') as f:
        f.write(str(weights))


def load_weights():
    with open('Weights.txt', 'r') as f:
        return eval(f.read())


class Qtable(dict):

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)


class DeepQLearning:

    def __init__(self, state_size, action_size, n):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = None
        self.gamma = 0.99
        self.memory = []
        self.alpha = 0.1
        self.model = self.build_network()
        self.n = n

    def build_network(self):
        model = Sequential()
        model.add(layers.Dense(8, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.alpha))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, current_iteration_number):
        self.set_parameter(current_iteration_number)
        if random_with_possibility(self.epsilon):
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.model.predict(state)[0])
        return action

    def set_parameter(self, current_iteration_number):
        self.epsilon = (self.n - current_iteration_number) / self.n

    def update(self, batch_size):
        first_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in first_batch:
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward
            target = np.reshape(target, [1, 1])
            self.model.fit(state, target, epochs=1, verbose=0)

    def train(self, env):
        next_state, info = env.reset(seed=42)
        score = 0
        for i in range(self.n):
            while True:
                state = np.reshape(next_state, [1, self.state_size])
                action = self.get_action(state, i)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state_temp = np.reshape(next_state, [1, self.state_size])
                self.memorize(state, action, reward, next_state_temp, terminated or truncated)
                score += reward
                # env.render()

                if terminated or truncated:
                    print("iteration number: ", i)
                    print(score)
                    if i % 100 == 0 and i != 0:
                        print("step :", i)
                        self.update(int((i / 100) * 50))
                    score = 0
                    next_state, info = env.reset()
                    break
