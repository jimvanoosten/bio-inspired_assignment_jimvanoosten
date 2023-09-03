import gymnasium as gym
from QLearning import QLearning
from DeepQLearning import Agent

"""
-------------------------------------------
uncomment the below to train with QLearning
-------------------------------------------
"""
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
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

"""
-----------------------------------------------
test1
-----------------------------------------------
"""
# learning_rates = [0.001, 0.0005, 0.0003, 0.0001]
# reward_curves = []
#
# for learning_rate in learning_rates:
#     env = gym.make("LunarLander-v2", render_mode="rgb_array")  # , render_mode="human"
#     number_of_iterations = 1000
#     discovery_decay = 1 / number_of_iterations * 1.25
#     deep_agent = Agent(gamma=0.99, epsilon=1, episode_num=number_of_iterations, batch_size=64, n=4, eps_min=0.01,
#                        input_dimensions=8, learning_rate=learning_rate, eps_dec=discovery_decay)
#     deep_agent.train(env)
#     reward_curves.append(deep_agent.reward_memory)

"""
-----------------------------------------------
test2
-----------------------------------------------
"""
# env = gym.make("LunarLander-v2", render_mode="human")
# number_of_iterations = 5
# discovery_decay = 1 / number_of_iterations * 1.25
# new_agent = Agent(gamma=0.99, epsilon=0.01, episode_num=1, batch_size=64, n=4, eps_min=0.01,
#                   input_dimensions=8, learning_rate=0.001, eps_dec=discovery_decay)
#
# # Load the successful model
# new_agent.load("successful_DeepQtable.pth")
#
# num_render_episodes = 5
#
# for episode in range(num_render_episodes):
#     state, info = env.reset(seed=42)
#     score = 0
#     done = False
#
#     while not done:
#         action = new_agent.choose_action(state)
#         next_state, reward, done, _, _ = env.step(action)
#         state = next_state
#         score += reward
#
#         # Render the environment in human mode
#         env.render()
#
#     print(f"Episode: {episode + 1}, Score: {score}")
#
# # Close the environment after rendering
# env.close()
