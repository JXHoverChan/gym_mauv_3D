import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import gym
import gym_auv

from time import time, sleep
from gym_auv.envs.pathfollowing3d import PathFollowing3d
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import PPO2, DDPG
from pyglet.window import key

env_config = {
    "reward_ds": 1,
    "reward_speed_error": -0.08,
    "reward_cross_track_error": -100,
    "reward_vertical_track_error": -100,
    "reward_heading_error": -10,
    "reward_pitch_error": -10,
    "reward_rudderchange": -0.01,
    "t_step": 0.1,
    "cruise_speed": 1.5,
    "la_dist": 50,
    "min_reward": -500,
    "max_timestemps": 10000}

if __name__ == "__main__":
    #agent = PPO2.load(load_path="agent1/521216.pkl")
    env = PathFollowing3d(env_config)
    ax = env.path.plot_path(label="Path")
    for i in range(1000):
        obs = env.observe()
        #action = agent.predict(obs)[0]
        action = np.array([1,1,0])
        env.step(action)
    ax.plot3D(env.vessel.path_taken[:, 0], env.vessel.path_taken[:, 1], env.vessel.path_taken[:, 2], label="AUV trajectory")
    plt.show()
    #for i in range(12):
        #plt.plot(env.vessel.state_trajectory[:,i])
        #plt.show()