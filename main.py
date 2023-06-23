import gymnasium as gym
import numpy as np
from agents import QLearning
# from agents import SarsaQLearning, ApproximateQLearning, DeepQLearning

env = gym.make("LunarLander-v2")
# env = gym.make("LunarLander-v2", render_mode="human")

qlearning_agent = QLearning(5, flag=True)
qlearning_agent.train(env)

# sarsa_agent = SarsaQLearning(5, flag=True)
# sarsa_agent.train(env)
#
# approximate_agent = ApproximateQLearning(5, flag=True)
# approximate_agent.train(env)
#
# deep_agent = DeepQLearning(5, flag=True)
# deep_agent.train(env)

env.close()
