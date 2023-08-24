import gymnasium as gym
from QLearning import QLearning
from DeepQLearning import Agent

"""
-------------------------------------------
uncomment the below to train with QLearning
-------------------------------------------
"""
# env = gym.make("LunarLander-v2", render_mode="human")
# qlearning_agent = QLearning(5, flag=True)
# qlearning_agent.train(env)


"""
-----------------------------------------------
uncomment the below to train with DeepQLearning
-----------------------------------------------
"""
env = gym.make("LunarLander-v2", render_mode="rgb_array")  # , render_mode="human"
number_of_iterations = 2000
discovery_decay = 1 / number_of_iterations * 1.25
deep_agent = Agent(gamma=0.99, epsilon=1, episode_num=number_of_iterations, batch_size=64, n=4, eps_min=0.01,
                   input_dimensions=8, learning_rate=0.001, eps_dec=discovery_decay)
deep_agent.train(env)

env.close()

