import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

env2 = gym.make("Taxi-v3", render_mode="human")
observation, info = env2.reset(seed=42)
for _ in range(1000):
    action = env2.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env2.step(action)
    print(f"reward: {reward} for action {action}")

    if terminated or truncated:
        observation, info = env2.reset()
env2.close()
