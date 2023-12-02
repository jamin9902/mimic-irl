"""
This code provides a transition model for the discretized version of the MIMIC dataset.

In order to use this environment natively with gymnasium, the following steps must be taken:
1) add this file to the local gymnasium installation under .../gymnasium/envs/classic_control

2) add the line:

from gymnasium.envs.classic_control.mimic import MimicEnv

to .../gymnasium/envs/classic_control/__init__.py

3) add the lines:

register(
    id="mimic-v0",
    entry_point="gymnasium.envs.classic_control:MimicEnv",
    max_episode_steps=72,
)

to .../gymnasium/envs/__init__.py

4) Now the environment can be loaded with the line:

gymnasium.make("mimic-v0")

Note that this will only make the mimic transition model available through the local gymnasium installation and not any global ones.
"""

import gymnasium as gym
from gymnasium import spaces
import pickle

with open("p_transition","rb") as f:
    transition_tensor = pickle.load(f)
    
transition_probs = {(i,j):[transition_tensor[i,k,j] for k in range(100)] for i in range(100) for j in range(4)}

with open("counts",'rb') as f:
    cluster_counts = pickle.load(f)
    
cluster_probs = [i/sum(cluster_counts) for i in cluster_counts]

class MimicEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(100)
        self.num_envs=1
        self.transition_probs = transition_probs
        self.step_limit = 72
        
    def step(self, action):
        terminated = (self.steps > self.step_limit)
        
        #this logic could use some tinkering to play around with the transitions
        #currently, we just use a naive transition according to the probabilities calculated in HW3
        next_state = self.np_random.choice(100,p=transition_probs[(self.curr_state, action)])

        #the calculation of rewards could also use some tinkering
        #currently, we use a naive reward equal to the probability of the transition
        reward = transition_probs[(self.curr_state, action)][next_state]
        self.curr_state = next_state
        return self.curr_state, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_state = self.np_random.choice(100,p=cluster_probs)
        self.steps = 0
        return self.curr_state, {}

