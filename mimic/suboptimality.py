import pandas as pd
import numpy as np
import random

# Observational ambiguity - add variance according to normal distribution to each column entry with probability p
# Normal distribution with mean 0 and standard deviation of column multiplied by std_multiplier
def observation_ambiguity(df, columns, std_multiplier, p_suboptimal):
    suboptimal_df = df.copy()
    suboptimal_std = (suboptimal_df[columns]).std()
    for i in columns:
        column = suboptimal_df[i]
        column_std = suboptimal_std[i]
        noise = np.random.normal(0, (std_multiplier * column_std), len(column))
        for j in range(len(noise)):
            if random.random() > p_suboptimal:
                noise[j] = 0
        suboptimal_df[i] = suboptimal_df[i] + noise
    return suboptimal_df

# Action ambiguity - randomly change action for each entry with probability p
def action_ambiguity(df, p_suboptimal):
    suboptimal_df = df.copy()
    for index, _ in suboptimal_df.iterrows():
        if random.random() > p_suboptimal:
            suboptimal_df.loc[index, 'action'] = random.randint(0, 3)
    return suboptimal_df

# Static occlusion - remove all instances of target states
def static_occlusion(df, target_states):
    suboptimal_df = df.copy()
    for i in target_states:
        suboptimal_df = suboptimal_df[suboptimal_df.cluster != i]
    return suboptimal_df

# Dynamic occlusion - remove each entry with probability p
def dynamic_occlusion(df, p_suboptimal):
    suboptimal_df = df.copy()
    indices = df.index
    num_drop = round(p_suboptimal * len(indices))
    drop_indices = np.random.choice(indices, num_drop, replace=False)
    suboptimal_df = suboptimal_df.drop(drop_indices)

