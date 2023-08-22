import gymnasium as gym
from QLearning import QLearning
from DeepQLearning import DeepQLearning

env = gym.make("LunarLander-v2")
# env = gym.make("LunarLander-v2", render_mode="human")

# qlearning_agent = QLearning(5, flag=True)
# qlearning_agent.train(env)

deep_agent = DeepQLearning(env.observation_space.shape[0], env.action_space.n, 1000)
deep_agent.train(env)

env.close()
