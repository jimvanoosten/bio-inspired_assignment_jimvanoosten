import gymnasium as gym

environment_name = "LunarLander-v2"

env = gym.make(environment_name, render_mode="human")
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info, a = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

# env = gym.make(environment_name, render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#
#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()


