"""
Evaluate Policies
"""

import gymnasium as gym
import numpy as np

# Mountain car environment
env = gym.make('MountainCar-v0')

# Discretize the state space into .1 increments for position and .01 increments for velocity 
# Total state space consisting of 19*15 = 285 states
def discretize(state):
    temp = (state - env.observation_space.low) * np.array([10,100])
    return np.round(temp, 0).astype(int)

# Extract reward from policies
def extract_reward(env, policy):
    found_goal = False
    total_reward = 0
    curr = discretize(env.reset()[0])
    while not found_goal:
        position = curr[0]
        velocity = curr[1]
        action = policy[(position * 15) + velocity]
        new_state, reward, found_goal, _, _ = env.step(action)
        if found_goal:
            reward = 100
        total_reward += reward
        curr = discretize(new_state)
    return total_reward

# Average reward across n trials
def average_reward(env, policy, n):
    sum = 0
    for _ in range(n):
        sum += extract_reward(env, policy)
    return (sum/n)
