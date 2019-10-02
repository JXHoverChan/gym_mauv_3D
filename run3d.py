import os
import sys
import subprocess
import numpy as np
from time import time, sleep
import argparse
import tensorflow as tf
import gym
import gym_auv

from gym_auv.envs.pathfollowing3d import PathFollowing3d
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import PPO2, DDPG


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


def create_env():
    env = PathFollowing3d(env_config)
    return env

if __name__ == '__main__':
    num_cpu = 1
    vec_env = DummyVecEnv([lambda: create_env()])

    hyperparams = {
        'n_steps': 1024,
        'nminibatches': 32,
        'lam': 0.98,
        'gamma': 0.999,
        'noptepochs': 4,
        'ent_coef': 0.01
    }
    
    agent = PPO2(MlpPolicy, vec_env, verbose=1, **hyperparams)
    
    print('Training {} agent on "{}"'.format('PPO', "Pathfollowing3D"))

    n_updates = 0
    def callback(_locals, _globals):
        global n_updates

        total_t_steps = _locals['self'].get_env().get_attr('total_t_steps')[0]*num_cpu
        agent_filepath = os.path.join(agent_folder, str(total_t_steps) + '.pkl')
        _locals['self'].save(agent_filepath)

        if (n_updates % 25 == 0 and args.mp):
            cmd = 'python run.py enjoy {} --agent "{}" --video-dir "{}" --video-name "{}" --length {}' .format(
                args.env, agent_filepath, video_folder, args.env + '-' + str(total_t_steps), video_length
            )
            subprocess.Popen(cmd)
    
        n_updates += 1
        
    agent.learn(
        total_timesteps=10000000, 
        tb_log_name='log',
        callback=callback
    )
