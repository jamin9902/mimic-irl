import pandas as pd
import numpy as np
import random

# Observational ambiguity - add variance according to normal distribution to each column with probability p 
def observation_ambiguity(df, columns, std_multiplier, p_suboptimal):
    suboptimal_df = df
    suboptimal_std = (suboptimal_df[columns]).std()

    for i in columns:
        column = suboptimal_df[i]
        column_std = suboptimal_std[i]
        noise = np.random.normal(0, (std_multiplier * column_std), len(column))
        for j in range(len(noise)):
            if random.random() > p_suboptimal:
                noise[j] = 0
                
    return suboptimal_df

# Action ambiguity - randomly change action for each entry with probability p
def action_ambiguity(df, p_suboptimal):
    return

# Static occlusion - remove all instances of target states
def static_occlusion(df, target_states):
    return 

# Dynamic occlusion - remove each entry with probability p
def dynamic_occlusion(df, p_suboptimal):
    return
