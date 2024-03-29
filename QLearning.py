import random
import matplotlib.pyplot as plt
import numpy as np


def draw_chart(rewards, average_rewards, name, lr):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Individual Episode Rewards', color='blue', alpha=0.5)
    plt.plot(average_rewards, label='Average Rewards (100 Episodes)', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title(f'{name} (Learning Rate: {lr})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(len(rewards), step=1000))
    plt.savefig(f'{name}_{lr}.png')
    plt.show()


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


class Qtable(dict):

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)


class QLearning:

    def __init__(self, n, flag=False):
        self.flag = flag  # boolean that determines if the epsilon value should be set to 0
        self.n = n  # number of iterations for training
        self.alpha = 0.1  # learning rate, controlling the weight given to new information
        self.gamma = 0.99  # discount factor, determining the importance of future rewards
        self.epsilon = None  # exploration rate, controlling the balance between exploration (choosing random actions)
        # and exploitation (choosing actions based on the learned Q-values)

        self.QValues = Qtable()  # instance of Qtable class, used to store Q-values
        self.QValues = load_table()
        self.action = [0, 1, 2, 3]  # list containing possible actions

    def get_q_values(self, state, action):
        try:
            return self.QValues[(state, action)]
        except KeyError:
            return 0

    def compute_value_from_q_values(self, state):
        maximum = self.get_q_values(state, self.action[0])
        for i in self.action:
            if self.get_q_values(state, i) > maximum:
                maximum = self.get_q_values(state, i)
        return maximum

    def compute_action_from_q_values(self, state):
        maximum = self.compute_value_from_q_values(state)
        for i in self.action:
            if self.get_q_values(state, i) == maximum:
                return i

    def get_action(self, state, current_iteration_number):
        self.set_parameter(current_iteration_number)
        if random_with_possibility(self.epsilon):
            action = random.choice(self.action)
        else:
            action = self.compute_action_from_q_values(state)
        return action

    def update(self, state, action, next_state, reward, current_iteration_number, terminated, truncated):
        if terminated or truncated:
            self.QValues[(state, action)] = self.get_q_values(state, action) + self.alpha * \
                                            (reward - self.get_q_values(state, action))
        else:
            self.QValues[(state, action)] = self.get_q_values(state, action) + self.alpha * \
                                            (reward + (self.gamma * self.compute_value_from_q_values(next_state)) -
                                             self.get_q_values(state, action))

    def set_parameter(self, current_iteration_number):
        if self.flag:
            self.epsilon = 0
        else:
            self.epsilon = (self.n - current_iteration_number) / self.n

    def train(self, env):
        iterations, rewards, average_rewards = list(), list(), list()
        next_state, info = env.reset(seed=42)
        score = 0
        for i in range(self.n):
            while True:
                state = next_state
                state = discretize_state(state)
                action = self.get_action(state, i + 1)
                next_state, reward, terminated, truncated, info = env.step(action)
                score += reward
                next_state_temp = discretize_state(next_state)
                self.update(state, action, next_state_temp, reward, i, terminated, truncated)
                env.render()

                if terminated or truncated:
                    iterations.append(i)
                    rewards.append(score)
                    average_reward = np.mean(rewards[-100:])
                    average_rewards.append(average_reward)
                    print("iteration number: ", i)
                    print(score)
                    print(average_reward)
                    # if score > 200:
                    #     print(f"You have found a solution after {i} iterations!")
                    #     return  # stop the training process if score > 200 is reached
                    score = 0
                    next_state, info = env.reset()
                    break

            if i == self.n - 1:
                save_table(self.QValues)
                print("Qtable saved as Qtable.txt")

        draw_chart(rewards, average_rewards, "Q-learning", self.alpha)
