"""
Expert Trajectories 
"""

import gymnasium as gym
import numpy as np
import pickle
import random

# Load huggingface expert trajectories
with open("continuous", "rb") as t:
    trajectories = pickle.load(t)

# Mountain car environment
env = gym.make('MountainCar-v0')

# Discretize the state space into .1 increments for position and .01 increments for velocity 
# Total state space consisting of 19*15 = 285 states
def discretize(state):
    temp = (state - env.observation_space.low) * np.array([10,100])
    return np.round(temp, 0).astype(int)

# Discretize trajectories
def discretize_trajectories(trajectories):
    discretized_trajectories = []
    for traj in trajectories: 
        discretized_traj = []
        for t in traj:
            state = t[0]
            discrete_state = discretize(state)
            discretized_traj.append((discrete_state, t[1]))
        discretized_trajectories.append(discretized_traj)
    return discretized_trajectories