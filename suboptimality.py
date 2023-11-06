import gymnasium as gym
import numpy as np
import random

# Observational Ambiguity - add variance to state for each state-action pair with probability p 
def observation_ambiguity(trajectory, p_suboptimal):
    return 


# Action ambiguity - randomly change action for each state-action pair with probability p
def action_ambiguity(trajectory, env, p_suboptimal):
    for i in range(len(trajectory)):
        if random.random() < p_suboptimal:
                action = np.random.randint(0, env.action_space.n)
                lst = list(trajectory[i])
                lst[1] = action
                trajectory[i] = tuple(lst)
    return trajectory

# Static occlusion - remove all instances of target states
def static_occlusion(trajectory, target_states):
    for t in trajectory:
        if t[0] in target_states:
            trajectory.remove(t)
    return trajectory

# Dynamic occlusion - remove each state-action pair with probability p
def dynamic_occlusion(trajectory, p_suboptimal):
    for t in trajectory:
        if random.random() < p_suboptimal:
            trajectory.remove(t)
    return trajectory
