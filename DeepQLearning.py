import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import array
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Network(nn.Module):
    def __init__(self, learning_rate, input_dimensions, layer1_dimensions, layer2_dimensions, n):
        super(Network, self).__init__()
        self.input_dimensions = input_dimensions
        self.layer1_dimensions = layer1_dimensions
        self.layer2_dimensions = layer2_dimensions
        self.n = n
        self.entry_layer = nn.Linear(self.input_dimensions, self.layer1_dimensions)
        self.middle_layer = nn.Linear(self.layer1_dimensions, self.layer2_dimensions)
        self.exit_layer = nn.Linear(self.layer2_dimensions, self.n)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def get_q_value(self, state):
        x = functional.relu(self.entry_layer(state))
        x = functional.relu(self.middle_layer(x))
        actions = self.exit_layer(x)

        return actions

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


def draw_chart(rewards, name, sort):
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.scatter(np.arange(len(rewards)), rewards, c='r', marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(sort)
    plt.show()
    ax.figure.savefig(f'{name}.png')


class Agent:
    def __init__(self, gamma, epsilon, episode_num, learning_rate, input_dimensions, batch_size, n, memory_size=200000,
                 eps_min=0.01,
                 eps_dec=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode_num = episode_num
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n)]
        self.batch_size = batch_size
        self.memory_pointer = 0

        self.state_memory = [[] for i in range(self.memory_size)]
        self.next_state_memory = [[] for i in range(self.memory_size)]
        self.reward_memory = [0 for i in range(self.memory_size)]
        self.action_memory = [0 for i in range(self.memory_size)]
        self.terminal_memory = [False for i in range(self.memory_size)]

        self.network = Network(learning_rate=self.learning_rate, n=n, input_dimensions=input_dimensions,
                               layer1_dimensions=256, layer2_dimensions=256)

    def memorize(self, state, action, reward, next_state, done):
        index = self.memory_pointer % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.memory_pointer += 1

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.tensor(state).to(self.network.device)
            actions = self.network.get_q_value(state)
            return torch.argmax(actions).item()
        else:
            return np.random.choice(self.action_space)

    def learn(self):
        if self.memory_pointer < self.batch_size:
            return

        self.network.optimizer.zero_grad()

        max_mem = min(self.memory_pointer, self.memory_size)
        batch = [np.random.choice(max_mem) for _ in range(self.batch_size)]
        batch_index = [i for i in range(self.batch_size)]

        state_memory_elements = [array.array('d', self.state_memory[index]) for index in batch]
        next_state_memory_elements = [array.array('d', self.next_state_memory[index]) for index in batch]
        reward_memory_elements = array.array('d', [self.reward_memory[index] for index in batch])
        terminal_memory_elements = [int(self.terminal_memory[index]) for index in batch]

        state_batch = torch.tensor(state_memory_elements).to(self.network.device)
        next_state_batch = torch.tensor(next_state_memory_elements).to(self.network.device)
        reward_batch = torch.tensor(reward_memory_elements).to(self.network.device)
        terminal_batch = torch.tensor(terminal_memory_elements).to(self.network.device)

        action_batch = [self.action_memory[index] for index in batch]

        Q_network = self.network.get_q_value(state_batch)[batch_index, action_batch]
        q_next = self.network.get_q_value(next_state_batch)
        q_target = reward_batch + self.gamma * (1 - terminal_batch) * torch.max(q_next, dim=1)[0]

        loss = self.network.loss(q_target, Q_network).to(self.network.device)
        loss.backward()
        self.network.optimizer.step()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save(self, path):
        self.network.save(path)

    def load(self, path):
        self.network.load(path)

    def train(self, env):
        state, info = env.reset(seed=2)

        # self.load("DeepQtable.pth")
        scores, eps_history, avg_scores = [], [], []
        for index, _ in enumerate(range(self.episode_num)):
            state, info = env.reset(seed=1)
            score = 0
            while True:
                state, score, done = self.go_to_next_state(score, env, state)
                if done:
                    break

            self.decrement_epsilon()
            scores.append(score)
            eps_history.append(self.epsilon)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            print('episode', index, 'score %.2f' % score, 'average score %.2f' % avg_score,
                  'epsilon %.2f' % self.epsilon)
            if avg_score >= 200:
                self.save('successful_DeepQtable.pth')
                break

        draw_chart(scores, "DeepQLearning", "Reward")
        draw_chart(avg_scores, "DeepQLearning", "Average Reward")
        self.save('DeepQtable.pth')
        env.close()

    def go_to_next_state(self, score, env, state):
        action = self.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        score += reward
        next_state = next_state
        self.memorize(state, action, reward, next_state, terminated | truncated)

        self.learn()
        if terminated or truncated:
            return next_state, score, True

        return next_state, score, False
