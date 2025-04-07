import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_auv
import os

import pandas as pd
from gym_auv.utils.controllers import PI, PID
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


if __name__ == "__main__":
    experiment_dir, agent_path, scenario = parse_experiment_info()
    env = gym.make("PathColav3d-v0", scenario=scenario)
    agent = PPO.load(agent_path)
    sim_df = simulate_environment_multi_vessels(env, agent)
    sim_df.to_csv(r'simdata_m.csv')
    # sim_df = pd.read_csv(r'simdata_m.csv')  # Reload to ensure we have the latest data
    calculate_IAE_multi_vessels(sim_df, env.num_vessels)
    plot_attitude_multi_vessels(sim_df, env.num_vessels)
    plot_velocity_multi_vessels(sim_df, env.num_vessels)
    plot_angular_velocity_vessels(sim_df, env.num_vessels)
    plot_control_inputs_multi_vessels([sim_df], env.num_vessels)
    plot_control_errors_multi_vessels([sim_df], env.num_vessels)
    plot_multiple_3d(env, sim_df, env.num_vessels)
    plot_current_data_multi_vessels(sim_df, env.num_vessels)

