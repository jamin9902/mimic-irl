"""
Behavioral Cloning (BC) Model
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pandas as pd

# Number of states and actions in environment
NUM_STATES = 285
NUM_ACTIONS = 3

# Converts position-velocity trajectories to linear state space
def convert_trajectories(trajectories):
    states = []
    actions = []
    for traj in trajectories:
        for t in traj:
            position = t[0][0]
            velocity = t[0][1]
            action = t[1]
            state_number = (position * 15) + velocity
            states.append(state_number)
            actions.append(action)
    return (states, actions)

# One-hot encoders for state and action space
state_encoder = OneHotEncoder(sparse=False, categories= [np.arange(NUM_STATES)])
action_encoder = OneHotEncoder(sparse=False, categories= [np.arange(NUM_ACTIONS)])

# BC model
def model(trajectories):
    # Read in trajectories
    states, actions = trajectories
    states_onehot = state_encoder.fit_transform(np.array(states).reshape(-1, 1))
    actions_onehot = action_encoder.fit_transform(np.array(actions).reshape(-1, 1)) 

    # Define neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(states_onehot.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(actions_onehot.shape[1], activation='softmax')  # Output layer with softmax for discrete actions
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

    # Train model
    model.fit(states_onehot, actions_onehot,  epochs=5, batch_size=128)

    # Evaluate model
    test_loss = model.evaluate(states_onehot, actions_onehot)
    print("Test Loss:", test_loss)

    return model

# Extract policy
def policy(model):
    policy = np.argmax(model.predict(state_encoder.transform(np.arange(NUM_STATES).reshape(-1, 1))), axis =1).tolist()
    return policy
