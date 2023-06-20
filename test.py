# import gymnasium as gym
#
# env = gym.make("LunarLander-v2", render_mode="human")
#
# episodes = 10
# for episode in range(1, episodes + 1):
#     observation, info = env.reset(seed=42)
#     terminated = False
#     truncated = False
#     score = 0
#
#     while not terminated or truncated:
#         env.render()
#         action = env.action_space.sample()  # this is where you would insert your policy
#         observation, reward, terminated, truncated, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
#
# env.close()

import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
