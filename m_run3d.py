import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_auv
import os

import pandas as pd
from gym_auv.utils.controllers import PI, PID
from gym_auv.utils.DQACA import DQACA
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


if __name__ == "__main__":
    # For DQACA Initialization
    params = {
    'alpha': 1,
    'beta': 10,
    'rho': 0.1,
    'Q': 100,
    'max_iter': 2000,
    'ant_count': 51,
    'epsilon_0': 0.1
    }

    # 示例数据（三维坐标）
    start_points = np.array([[35,50,35], [35,50,35], [35,50,35], [35,50,35]])  # 4个AUV的起始点
    task_points = np.array([
        [30, 194, 140], [35, 156, 94], [121, 94, 99], [13, 32, 62], [185, 141, 104],
        [32, 148, 85], [113, 40, 117], [171, 11, 88], [24, 128, 150], [89, 34, 104],
        [78, 5, 25], [84, 22, 145], [2, 120, 57], [90, 96, 53], [75, 152, 146],
        [130, 14, 133], [114, 8, 125], [82, 144, 61], [198, 128, 120], [200, 183, 127]
    ])  # 20个任务点
    resources = [6, 6, 4, 4]  # 每个AUV的任务资源限制

    total_tasks = len(task_points)
    if sum(resources) < total_tasks:
        print("任务资源不足，无法分配所有任务点。")
        exit()
    else:
        print("任务资源足够，开始运行算法。")
    solver = DQACA(start_points, task_points, resources, params)
    best_paths, best_distance, total, best = solver.run()
    print(f"最优总距离: {best_distance}")
    for i, path in enumerate(best_paths):
        point_path = [solver.all_points[n] for n in path]
        print(f"AUV {i+1} 的路径:")
        for p in point_path:
            print(f"({p[0]}, {p[1]}, {p[2]})")
        print()
    solver.data_plot()
    solver.route_plot()
    waypoints = []
    for i, path in enumerate(best_paths):
        point_path = [solver.all_points[n] for n in path]
        waypoints.append(point_path)
    print("所有AUV的路径:")
    for wp in waypoints:
        print(wp)
    experiment_dir, agent_path, scenario = parse_experiment_info()
    env = gym.make("PathColav3d-v0", scenario=scenario, waypoints=waypoints)
    agent = PPO.load(agent_path)
    sim_df = simulate_environment_multi_vessels(env, agent)
    sim_df.to_csv(r'simdata_m.csv')
    calculate_IAE_multi_vessels(sim_df, env.num_vessels)
    plot_attitude_multi_vessels(sim_df, env.num_vessels)
    plot_velocity_multi_vessels(sim_df, env.num_vessels)
    plot_angular_velocity_vessels(sim_df, env.num_vessels)
    plot_control_inputs_multi_vessels([sim_df], env.num_vessels)
    plot_control_errors_multi_vessels([sim_df], env.num_vessels)
    plot_multiple_3d(env, sim_df, env.num_vessels)
    plot_current_data_multi_vessels(sim_df, env.num_vessels)

