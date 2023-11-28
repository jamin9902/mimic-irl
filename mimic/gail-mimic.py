import numpy as np
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper

SEED = 42

#need to add mimic.py to gymnasium local installation to use this env
env = make_vec_env(
    "mimic-v0",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],
)

from imitation.data.types import Trajectory
import pickle

with open("mimic-trajectories","rb") as f:
    trajectories = pickle.load(f)

rollouts = []
#turn trajectories into (state,action) pairs
for trajectory in trajectories:
    obs = np.array(list(trajectory.states()))
    acts = np.array([i for (_,i,_) in trajectory.transitions()])
    rollouts.append(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# This is a learner that uses the proximal policy optimization algorithm and we will it use to compute the policy update steps of the GAIL algorithm
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)

# This is a neural network that takes a batch of (state, action, next_state) triples and calculates the associated rewards and we will use it to compute the discriminator update steps of the GAIL algorithm
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

# imitation implementation of GAIL 
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(800_000)

#extract policy
policy = [learner.predict(i,deterministic=True).item() for i in range(100)]
