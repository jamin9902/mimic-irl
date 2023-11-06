import gymnasium as gym
import numpy as np
import pickle
import random
import pandas as pd


with open("disc_coarse", "rb") as fp:
    trajectories = pickle.load(fp)